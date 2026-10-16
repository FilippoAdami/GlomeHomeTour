"""Class-agnostic refinement of existing mesh labels from frozen multi-view evidence."""
from __future__ import annotations

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _sum_block_unaries(unary, component, count):
    import torch

    if unary.device.type == 'cpu':
        costs = torch.zeros((count,unary.shape[1]),dtype=torch.float64,device=unary.device)
        for j in range(unary.shape[1]):
            costs[:,j].index_add_(0,torch.as_tensor(component),unary[:,j].double())
        return costs
    # Bounded segments avoid serializing millions of FP64 atomics into one surface.
    order = np.argsort(component,kind='stable')
    sorted_blocks = component[order]
    starts = np.unique(np.r_[0,np.flatnonzero(np.diff(sorted_blocks))+1,np.arange(0,len(order),512)])
    lengths = np.diff(np.r_[starts,len(order)])
    chunks_per_block = np.bincount(sorted_blocks[starts],minlength=count)
    ordered = unary[torch.as_tensor(order,device=unary.device)].double()
    partial = torch.segment_reduce(ordered,'sum',lengths=torch.as_tensor(lengths,device=unary.device))
    return torch.segment_reduce(partial,'sum',lengths=torch.as_tensor(chunks_per_block,device=unary.device))


def _component_orientation_penalties(reference, labels, areas, normals, u, v, component, anchors, device):
    import torch

    vocabulary=np.unique(labels[labels>0])
    valid=(reference>0)&(areas>0)
    same=valid[u]&valid[v]&(reference[u]==reference[v])
    count,regions=connected_components(csr_matrix((np.ones(same.sum()),(u[same],v[same])),
        shape=(len(labels),len(labels))),directed=False)
    good,first=np.unique(regions[anchors],return_index=True)
    model_ids=reference[np.flatnonzero(anchors)[first]]
    lookup=np.full(count,-1);lookup[good]=np.arange(len(good))
    fit=valid&(lookup[regions]>=0)
    group=lookup[regions[fit]];weight=areas[fit];normal=normals[fit]
    model_area=np.bincount(group,weights=weight,minlength=len(good))
    means=np.column_stack([np.bincount(group,weights=weight*normal[:,j],minlength=len(good))
        for j in range(3)])/model_area[:,None]
    second=np.stack([np.bincount(group,weights=weight*normal[:,i]*normal[:,j],minlength=len(good))
        for i in range(3) for j in range(3)],axis=1).reshape(-1,3,3)/model_area[:,None,None]
    live=(areas[u]>0)&(areas[v]>0)
    # Measured tessellation noise keeps fitted planes tolerant of mesh roughness.
    noise=max(float(np.median(1-np.clip((normals[u[live]]*normals[v[live]]).sum(axis=1),-1,1))) if live.any() else 0.,1e-6)
    precision=np.linalg.inv(second-means[:,:,None]*means[:,None,:]+noise*np.eye(3))
    query_weight=areas*(labels>0)
    query_area=np.bincount(component,weights=query_weight)
    query_mean=np.column_stack([np.bincount(component,weights=query_weight*normals[:,j])
        for j in range(3)])/np.maximum(query_area[:,None],1e-30)
    query=np.flatnonzero(query_area>0)
    dtype=torch.float32 if device.startswith('cuda') else torch.float64
    query_tensor=torch.as_tensor(query_mean[query],dtype=dtype,device=device)
    means=torch.as_tensor(means,dtype=dtype,device=device)
    precision=torch.as_tensor(precision,dtype=dtype,device=device)
    costs=torch.zeros((len(query_area),len(vocabulary)),dtype=dtype,device=device)
    query_index=torch.as_tensor(query,device=device)
    for j,cid in enumerate(vocabulary):
        matching=np.flatnonzero(model_ids==cid)
        if not len(matching):
            continue
        models=torch.as_tensor(matching,device=device)
        delta=query_tensor[:,None,:]-means[models][None,:,:]
        distance=torch.einsum('qmi,mij,qmj->qm',delta,precision[models],delta)
        # Compare whole-component means; retain every supported instance orientation.
        costs[query_index,j]=(distance.min(dim=1).values*.5-1.5).clamp_min(0)
    return costs[torch.as_tensor(component,device=device)]


def correct_confident_faces(
    initial_labels: np.ndarray,
    priors: csr_matrix,
    independent_views: csr_matrix,
    source_ids: np.ndarray,
    min_views: int = 3,
    min_confidence: float = 0.75,
) -> np.ndarray:
    """Correct only corroborated faces, preserving the supplied label vocabulary.

    Priors outside that vocabulary remain in the probability denominator: omitting
    competing evidence must not turn a weak in-vocabulary prediction into certainty.
    Each view count must represent independent camera groups, not pixels or frames;
    accumulated quality must also reach that many effective, unit-quality views.
    """
    source_ids = np.asarray(source_ids)
    if initial_labels.ndim != 1 or source_ids.ndim != 1:
        raise ValueError('labels and source IDs must be one-dimensional')
    if priors.shape != (len(initial_labels), len(source_ids)) or independent_views.shape != priors.shape:
        raise ValueError('evidence dimensions must match faces and source IDs')
    if (not np.issubdtype(source_ids.dtype, np.integer) or
        len(np.unique(source_ids)) != len(source_ids) or np.any(source_ids <= 0)):
        raise ValueError('source IDs must be unique positive integers')
    if min_views < 1 or not 0 < min_confidence <= 1:
        raise ValueError('invalid evidence thresholds')
    if (np.any(~np.isfinite(priors.data)) or np.any(priors.data < 0) or
        np.any(~np.isfinite(independent_views.data)) or np.any(independent_views.data < 0)):
        raise ValueError('evidence must be finite and nonnegative')

    result = initial_labels.copy()
    columns = np.flatnonzero(np.isin(source_ids, np.unique(initial_labels[initial_labels > 0])))
    if len(columns) == 0:
        return result
    allowed = priors[:, columns].tocsr()
    winners = np.asarray(allowed.argmax(axis=1)).ravel()
    rows = np.arange(len(initial_labels))
    total = np.asarray(priors.sum(axis=1)).ravel()
    weight = np.asarray(allowed[rows, winners]).ravel()
    confidence = weight / np.maximum(total, 1e-12)
    counts = np.asarray(independent_views[rows, columns[winners]]).ravel()
    accepted = (counts >= min_views) & (weight >= min_views) & (confidence >= min_confidence)
    result[accepted] = source_ids[columns[winners[accepted]]]
    return result


def refine_graph_labels(
    initial_labels: np.ndarray,
    priors: csr_matrix,
    source_ids: np.ndarray,
    face_areas: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_lengths: np.ndarray,
    boundary_scale_m: float,
    max_iterations: int = 100,
    device: str | None = None,
    geometry_reference_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Minimize an area/length-weighted Potts energy without semantic-role rules.

    Frozen source distributions protect observed boundaries, including sources not
    in the output vocabulary. Unsupported sources give no positive unary evidence.
    A weak initialization prior protects corroborated components, including parts
    with a strongly observed source outside the vocabulary. Unsupported components
    may relax through observed geometry. Unknown geometry stays unknown. Surface blocks
    and regions defined by the priors can move together to escape local minima.
    Each parallel update is an independent set and decreases the actual energy.
    An optional earlier label checkpoint supplies independent component orientation
    models. Only components corroborated by frozen observations train those models;
    their soft compatibility term compares whole components, preserving local roughness.
    """
    import torch

    n = len(initial_labels)
    if geometry_reference_labels is not None and (geometry_reference_labels.shape != (n,) or
        not np.issubdtype(geometry_reference_labels.dtype,np.integer)):
        raise ValueError('geometry reference must contain one integer label per face')
    if priors.shape != (n, len(source_ids)) or normals.shape != (n, 3) or rgb.shape != (n, 3):
        raise ValueError('evidence and geometry dimensions disagree')
    if (not np.issubdtype(source_ids.dtype, np.integer) or
        len(np.unique(source_ids)) != len(source_ids) or np.any(source_ids <= 0)):
        raise ValueError('source IDs must be unique positive integers')
    if (len(face_areas) != n or len(edge_u) != len(edge_v) or len(edge_lengths) != len(edge_u) or
        boundary_scale_m <= 0 or not np.isfinite(boundary_scale_m) or max_iterations < 1):
        raise ValueError('invalid areas, edges or stopping conditions')
    if (np.any(~np.isfinite(priors.data)) or np.any(priors.data < 0) or
        np.any(~np.isfinite(face_areas)) or np.any(face_areas < 0) or
        np.any(~np.isfinite(normals)) or np.any(~np.isfinite(rgb)) or
        np.any(~np.isfinite(edge_lengths)) or np.any(edge_lengths < 0)):
        raise ValueError('evidence and geometry must be finite and nonnegative')
    if np.any(edge_u < 0) or np.any(edge_v < 0) or np.any(edge_u >= n) or np.any(edge_v >= n):
        raise ValueError('edge index outside mesh')
    vocabulary = np.unique(initial_labels[initial_labels > 0])
    if len(vocabulary) == 0:
        return initial_labels.copy(), {'converged': True, 'energies': [], 'changes': []}
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if device.startswith('cuda') else torch.float64
    raw = priors.tocoo()
    evidence = torch.zeros(priors.shape, dtype=dtype, device=device)
    evidence[torch.as_tensor(raw.row.astype(np.int64), device=device),
             torch.as_tensor(raw.col.astype(np.int64), device=device)] = torch.as_tensor(raw.data, dtype=dtype, device=device)
    total = evidence.sum(dim=1)
    positions = np.searchsorted(vocabulary, source_ids)
    recognized = positions < len(vocabulary)
    recognized[recognized] &= vocabulary[positions[recognized]] == source_ids[recognized]
    known = torch.zeros((n, len(vocabulary)), dtype=dtype, device=device)
    known[:, torch.as_tensor(positions[recognized], device=device)] = evidence[:, torch.as_tensor(np.flatnonzero(recognized), device=device)]
    known_geometry = (initial_labels > 0) & (face_areas > 0)
    observed_np = known_geometry & (np.asarray(priors.sum(axis=1)).ravel() > 0)
    active = torch.as_tensor(observed_np, device=device)
    initial = np.searchsorted(vocabulary, initial_labels).clip(0, len(vocabulary)-1)
    labels = torch.as_tensor(initial.astype(np.int64), device=device)
    rows = torch.arange(n, device=device)
    # One uniform pseudo-view regularizes sparse evidence; quality controls its influence.
    known_total = known.sum(dim=1)
    data_target = known.argmax(dim=1)
    data_support = known_geometry & ((known == known[rows,data_target,None]).sum(dim=1) == 1).cpu().numpy()
    data_support &= known_total.cpu().numpy() > 0
    known += 1 / len(vocabulary)
    probabilities = known / known.sum(dim=1, keepdim=True)
    area = torch.as_tensor(face_areas, dtype=dtype, device=device)
    raw_weight, raw_column = evidence.max(dim=1)
    raw_ids = source_ids[raw_column.cpu().numpy()]
    raw_weight_np = raw_weight.cpu().numpy()
    # Retaining an existing object requires less evidence than replacing its label.
    corroborating = (raw_ids == initial_labels) & (raw_weight_np > 0)
    foreign_support = ~np.isin(raw_ids,vocabulary) & (raw_weight_np >= 3)
    foreign_support &= (raw_weight/total.clamp_min(1e-12)).cpu().numpy() >= .75
    corroborating = known_geometry & (corroborating | foreign_support)
    same_initial = known_geometry[edge_u] & known_geometry[edge_v] & (initial_labels[edge_u] == initial_labels[edge_v])
    component_count, component = connected_components(csr_matrix(
        (np.ones(same_initial.sum()),(edge_u[same_initial],edge_v[same_initial])),shape=(n,n)),directed=False)
    protected = (np.bincount(component,weights=corroborating,minlength=component_count)>0)[component]
    initial_cost = .15*torch.as_tensor(protected,dtype=dtype,device=device)
    unary = -probabilities.log() * (known_total/(known_total+1))[:,None] + initial_cost[:,None]
    unary[rows,labels] -= initial_cost
    unary *= area[:,None]
    best_data = unary.argmin(dim=1)
    if geometry_reference_labels is not None:
        anchors=(geometry_reference_labels==raw_ids)&(raw_weight_np>=3)&(face_areas>0)
        anchors &= (raw_weight/total.clamp_min(1e-12)).cpu().numpy()>=.75
        if anchors.any():
            unary += area[:,None]*_component_orientation_penalties(geometry_reference_labels,
                initial_labels,face_areas,normals,edge_u,edge_v,component,anchors,device)
    u = torch.as_tensor(edge_u.astype(np.int64), device=device)
    v = torch.as_tensor(edge_v.astype(np.int64), device=device)
    normal = torch.as_tensor(normals, dtype=dtype, device=device)
    color = torch.as_tensor(rgb, dtype=dtype, device=device)
    angle = torch.acos((normal[u] * normal[v]).sum(dim=1).clamp(-1, 1))
    affinity = torch.exp(-angle.square() / (2 * np.radians(25)**2))
    affinity *= torch.exp(-(color[u]-color[v]).square().sum(dim=1) / (2 * .035**2))
    roots = (evidence / total.clamp_min(1e-12)[:, None]).sqrt()
    reliability = total / (total + 1)
    for start in range(0, len(edge_u), 262144):
        stop = start + 262144
        a, b = u[start:stop], v[start:stop]
        overlap = (roots[a] * roots[b]).sum(dim=1).clamp(0, 1)
        affinity[start:stop] *= 1 - reliability[a]*reliability[b]*(1-overlap)
    weight = affinity * torch.as_tensor(edge_lengths, dtype=dtype, device=device) * boundary_scale_m
    fixed_or_observed = torch.as_tensor(known_geometry,device=device)
    weight *= fixed_or_observed[u] & fixed_or_observed[v]
    del evidence, roots, known, probabilities

    live = weight.cpu().numpy() > 0
    region_count, region = connected_components(csr_matrix(
        (np.ones(live.sum()),(edge_u[live],edge_v[live])),shape=(n,n)),directed=False)
    anchored = np.bincount(region,weights=observed_np,minlength=region_count)>0
    active_np = known_geometry & anchored[region]
    active = torch.as_tensor(active_np,device=device)

    # CSR-to-dense neighbor slots retain arbitrary manifold boundary valence.
    graph = csr_matrix((np.ones(2*len(edge_u)), (np.r_[edge_u,edge_v],np.r_[edge_v,edge_u])), shape=(n,n))
    degrees = np.diff(graph.indptr)
    if graph.nnz != 2*len(edge_u):
        raise ValueError('edges must be unique and undirected')
    degree = int(degrees.max(initial=0))
    neighbor_np = np.broadcast_to(np.arange(n)[:,None],(n,degree)).copy()
    edge_order = csr_matrix((np.r_[np.arange(len(edge_u))+1,np.arange(len(edge_u))+1],
        (np.r_[edge_u,edge_v],np.r_[edge_v,edge_u])), shape=(n,n))
    slot = np.arange(graph.nnz)-np.repeat(graph.indptr[:-1],degrees)
    row_np = np.repeat(np.arange(n),degrees)
    neighbor_np[row_np,slot] = graph.indices
    edge_np = np.full((n,degree),-1,np.int64)
    edge_np[row_np,slot] = edge_order.data-1
    neighbors = torch.as_tensor(neighbor_np,device=device)
    local_weight = torch.zeros((n,degree),dtype=dtype,device=device)
    valid = edge_np>=0
    local_weight[torch.as_tensor(valid,device=device)] = weight[torch.as_tensor(edge_np[valid],device=device)]

    def energy():
        return float((unary[rows,labels].double().sum() + (weight*(labels[u]!=labels[v])).double().sum()).item())

    strong_np = (affinity.cpu().numpy() >= .5) & known_geometry[edge_u] & known_geometry[edge_v]

    def select_blocks(gain, eligible, a, b):
        outranked_a = eligible[b] & ((gain[b] > gain[a]) | ((gain[b] == gain[a]) & (b < a)))
        outranked_b = eligible[a] & ((gain[a] > gain[b]) | ((gain[a] == gain[b]) & (a < b)))
        blocked = torch.zeros(len(gain),dtype=torch.long,device=device)
        blocked.index_add_(0,a,outranked_a.long())
        blocked.index_add_(0,b,outranked_b.long())
        return eligible & (blocked == 0)

    def move_surface_blocks(whole_fragment=False):
        labels_np = labels.cpu().numpy()
        connected = known_geometry[edge_u] & known_geometry[edge_v] & (edge_lengths > 0) if whole_fragment else strong_np
        same_surface = connected & (labels_np[edge_u] == labels_np[edge_v])
        count, component = connected_components(csr_matrix(
            (np.ones(same_surface.sum()), (edge_u[same_surface], edge_v[same_surface])),
            shape=(n,n)), directed=False)
        block = torch.as_tensor(component,device=device)
        k = len(vocabulary)
        costs = _sum_block_unaries(unary,component,count)
        crossing = (block[u] != block[v]) & (weight > 0)
        a,b = block[u[crossing]],block[v[crossing]]
        contact = torch.zeros_like(costs)
        keys = torch.cat((a*k+labels[v[crossing]],b*k+labels[u[crossing]]))
        contact.flatten().index_add_(0,keys,weight[crossing].double().repeat(2))
        current = torch.empty(count,dtype=torch.long,device=device)
        current[block] = labels
        block_rows = torch.arange(count,device=device)
        delta = costs-costs[block_rows,current,None]+contact[block_rows,current,None]-contact
        delta[contact == 0] = torch.inf
        delta[block_rows,current] = 0
        minimum = delta.min(dim=1).values
        # Break equal-energy label ties by neighboring face order, never by source ID.
        first_neighbor = torch.full((count,k),n,dtype=torch.long,device=device)
        first_neighbor.flatten().scatter_reduce_(0,keys,torch.cat((v[crossing],u[crossing])),reduce='amin')
        proposed = torch.where(delta == minimum[:,None],first_neighbor,n).argmin(dim=1)
        movable = torch.zeros(count,dtype=torch.long,device=device)
        movable.index_add_(0,block,active.long())
        block_area = torch.as_tensor(np.bincount(component,weights=face_areas,minlength=count),device=device)
        gain = -minimum
        eligible = (movable > 0) & (gain > block_area*1e-6)
        if whole_fragment:
            # A crease may split a false fragment; never collapse directly supported objects.
            own_support = known_geometry & (raw_ids == vocabulary[labels_np]) & (raw_weight_np > 0)
            supported = np.bincount(component,weights=own_support,minlength=count)>0
            eligible &= ~torch.as_tensor(supported,device=device)
        selected = select_blocks(gain,eligible,a,b)
        changed_faces = selected[block]
        labels[changed_faces] = proposed[block[changed_faces]]
        return int(changed_faces.sum().item())

    def move_prior_regions():
        targets = data_target.cpu().numpy()
        same = strong_np & data_support[edge_u] & data_support[edge_v] & (targets[edge_u] == targets[edge_v])
        count, component = connected_components(csr_matrix(
            (np.ones(same.sum()),(edge_u[same],edge_v[same])),shape=(n,n)),directed=False)
        block = torch.as_tensor(component,device=device)
        face_delta = unary[rows,data_target]-unary[rows,labels]
        delta = _sum_block_unaries(face_delta[:,None],component,count)[:,0]
        internal_cut = (block[u] == block[v]) & (labels[u] != labels[v]) & (weight > 0)
        delta.index_add_(0,block[u[internal_cut]],-weight[internal_cut].double())
        crossing = (block[u] != block[v]) & (weight > 0)
        a,b = block[u[crossing]],block[v[crossing]]
        old_cut = (labels[u[crossing]] != labels[v[crossing]]).double()
        edge_weight = weight[crossing].double()
        delta.index_add_(0,a,edge_weight*((data_target[u[crossing]] != labels[v[crossing]]).double()-old_cut))
        delta.index_add_(0,b,edge_weight*((data_target[v[crossing]] != labels[u[crossing]]).double()-old_cut))
        block_area = torch.as_tensor(np.bincount(component,weights=face_areas,minlength=count),device=device)
        supported = torch.as_tensor(np.bincount(component,weights=data_support,minlength=count)>0,device=device)
        selected = select_blocks(-delta,supported & (-delta > block_area*1e-6),a,b)
        changed_faces = selected[block] & (labels != data_target)
        labels[changed_faces] = data_target[changed_faces]
        return int(changed_faces.sum().item())

    energies, changes, move_types, iteration_seconds = [energy()], [], [], []
    converged = False
    for _ in range(max_iterations):
        started = time.perf_counter()
        adjacent_labels = labels[neighbors]
        candidates = torch.cat((labels[:,None],best_data[:,None],adjacent_labels),dim=1)
        costs = unary[rows[:,None],candidates].clone()
        for j in range(degree):
            costs += local_weight[:,j,None] * (candidates != adjacent_labels[:,j,None])
        minimum, chosen = costs.min(dim=1)
        gain = costs[:,0] - minimum
        proposed = candidates[rows,chosen]
        eligible = active & (gain > area*1e-6)
        if not bool(eligible.any()):
            changed = move_surface_blocks()
            if changed == 0:
                changed = move_prior_regions()
                if changed == 0:
                    changed = move_surface_blocks(whole_fragment=True)
                    if changed == 0:
                        converged = True
                        break
                    move_types.append('component_block')
                else:
                    move_types.append('prior_region')
            else:
                move_types.append('surface_block')
        else:
            blocked = torch.zeros(n,dtype=torch.bool,device=device)
            for j in range(degree):
                neighbor = neighbors[:,j]
                outranks = (gain[neighbor]>gain) | ((gain[neighbor]==gain)&(neighbor<rows))
                blocked |= (local_weight[:,j]>0) & eligible[neighbor] & outranks
            selected = eligible & ~blocked
            changed = int(selected.sum().item())
            if changed == 0:
                raise RuntimeError('independent update selection made no progress')
            labels[selected] = proposed[selected]
            move_types.append('face')
        current = energy()
        if current > energies[-1] + max(1e-8, abs(energies[-1])*1e-7):
            raise RuntimeError('refinement increased the joint energy')
        energies.append(current)
        changes.append(changed)
        iteration_seconds.append(time.perf_counter()-started)
        if device.startswith('cuda'):
            time.sleep(.01)  # Let the desktop compositor run between synchronized rounds.
    result = initial_labels.copy()
    result[known_geometry] = vocabulary[labels.cpu().numpy()[known_geometry]]
    return result, {'converged': converged, 'energies': energies, 'changes': changes,
                    'move_types': move_types, 'iteration_seconds': iteration_seconds,
                    'unobserved_active_faces': int((active_np & ~observed_np).sum()),
                    'initially_protected_faces': int((protected & known_geometry).sum()),
                    'device': device, 'boundary_scale_m': boundary_scale_m}
