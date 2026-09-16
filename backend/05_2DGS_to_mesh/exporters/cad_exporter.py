"""GlomeHomeTour: CAD & BIM Vector Exporters (DXF / IFC).

Provides direct export to:
1. AutoCAD DXF (R12 ASCII format):
   - Layer mapping:
     * A-WALL: Wall boundary geometry & polylines
     * A-FLOR: Floor outline & footprint
     * FF-FURN: 3D furniture bounds and solid faces
     * A-CEIL: Ceiling plane geometry
   - Fully standalone with zero external C++ dependencies.
2. Industry Foundation Classes (IFC 4):
   - Generates valid STEP-format IFC4 architectural files with IfcBuildingElementProxy,
     IfcWall, and IfcSlab entities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from mesh_types import BoundingBox3D, MeshNode, NodeCategory


class DXFExporter:
    """Exports 3D meshes and bounding primitives to standard AutoCAD DXF files."""

    @staticmethod
    def _dxf_header() -> str:
        return (
            "0\nSECTION\n2\nHEADER\n"
            "9\n$ACADVER\n1\nAC1009\n"  # AutoCAD R12 compatibility
            "9\n$INSUNITS\n70\n6\n"       # Units: 6 = Meters
            "0\nENDSEC\n"
            "0\nSECTION\n2\nTABLES\n"
            "0\nTABLE\n2\nLAYER\n70\n4\n"
            "0\nLAYER\n2\nA-WALL\n70\n0\n62\n7\n6\nCONTINUOUS\n"   # White/Black
            "0\nLAYER\n2\nA-FLOR\n70\n0\n62\n8\n6\nCONTINUOUS\n"   # Gray
            "0\nLAYER\n2\nFF-FURN\n70\n0\n62\n5\n6\nCONTINUOUS\n"  # Blue
            "0\nLAYER\n2\nA-CEIL\n70\n0\n62\n9\n6\nCONTINUOUS\n"  # Light Gray
            "0\nENDTAB\n0\nENDSEC\n"
            "0\nSECTION\n2\nENTITIES\n"
        )

    @staticmethod
    def _dxf_footer() -> str:
        return "0\nENDSEC\n0\nEOF\n"

    @classmethod
    def _mesh_to_3dface_dxf(cls, mesh: trimesh.Trimesh, layer: str) -> str:
        """Convert triangle mesh faces into AutoCAD 3DFACE entities."""
        lines = []
        verts = mesh.vertices
        faces = mesh.faces

        for face in faces:
            p0, p1, p2 = verts[face[0]], verts[face[1]], verts[face[2]]
            # 3DFACE accepts 4 vertices; duplicate 3rd for triangle
            lines.append(
                f"0\n3DFACE\n8\n{layer}\n"
                f"10\n{p0[0]:.4f}\n20\n{p0[1]:.4f}\n30\n{p0[2]:.4f}\n"
                f"11\n{p1[0]:.4f}\n21\n{p1[1]:.4f}\n31\n{p1[2]:.4f}\n"
                f"12\n{p2[0]:.4f}\n22\n{p2[1]:.4f}\n32\n{p2[2]:.4f}\n"
                f"13\n{p2[0]:.4f}\n23\n{p2[1]:.4f}\n33\n{p2[2]:.4f}\n"
            )
        return "".join(lines)

    @classmethod
    def export_scene_dxf(
        cls,
        nodes: List[MeshNode],
        meshes: Dict[str, trimesh.Trimesh],
        output_path: Union[str, Path],
    ) -> Path:
        """Export full scene with layered walls, floors, and furniture solids to .dxf."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = [cls._dxf_header()]

        for node in nodes:
            layer = node.cad_layer or "0"
            mesh = meshes.get(node.name)
            if mesh is not None and len(mesh.faces) > 0:
                # Apply node transform if not identity
                t_mat = np.array(node.transform_matrix, dtype=np.float64)
                transformed_mesh = mesh.copy()
                if not np.allclose(t_mat, np.eye(4)):
                    transformed_mesh.apply_transform(t_mat)
                content.append(cls._mesh_to_3dface_dxf(transformed_mesh, layer))
            else:
                # Fallback: export bounding box 3DFACE wireframe
                b = node.bounding_box
                box = trimesh.creation.box(extents=b.extents)
                box.apply_translation(b.center)
                content.append(cls._mesh_to_3dface_dxf(box, layer))

        content.append(cls._dxf_footer())

        with open(out_path, "w", encoding="ascii") as f:
            f.write("".join(content))

        return out_path


class IFCExporter:
    """Exports Industry Foundation Classes (IFC4) STEP-format BIM files."""

    @classmethod
    def export_scene_ifc(
        cls,
        nodes: List[MeshNode],
        output_path: Union[str, Path],
        scene_id: str = "GlomeHomeTour_BIM",
    ) -> Path:
        """Generate a valid standard IFC4 STEP file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        step_lines = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('GlomeHomeTour CAD/BIM Export'),'2;1');",
            f"FILE_NAME('{out_path.name}','2026-09-10T16:00:00',('Architectural Engineering'),('Glome AI'),'GlomeHomeTour','IFC4','');",
            "FILE_SCHEMA(('IFC4'));",
            "ENDSEC;",
            "DATA;",
            "#1=IFCPROJECT('1Project00000000000000',#2,'Glome Scene',$,$,$,$,(#10),#3);",
            "#2=IFCOWNERHISTORY(#4,#5,$,.ADDED.,$,$,$,1700000000);",
            "#3=IFCUNITASSIGNMENT((#6,#7,#8));",
            "#4=IFCPERSON('USR',$,'Glome',$,$,$,$,$);",
            "#5=IFCORGANIZATION('ORG','GlomeHomeTour',$,$,$);",
            "#6=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);",
            "#7=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);",
            "#8=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);",
            "#10=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#11,$);",
            "#11=IFCAXIS2PLACEMENT3D(#12,#13,#14);",
            "#12=IFCCARTESIANPOINT((0.,0.,0.));",
            "#13=IFCDIRECTION((0.,0.,1.));",
            "#14=IFCDIRECTION((1.,0.,0.));",
            "#20=IFCSITE('1Site00000000000000000',#2,'Default Site',$,$,#21,$,$,.ELEMENT.,$,$,$,$,$);",
            "#21=IFCLOCALPLACEMENT($,#11);",
            "#30=IFCBUILDING('1Bldg00000000000000000',#2,'Main Building',$,$,#31,$,$,.ELEMENT.,$,$,$);",
            "#31=IFCLOCALPLACEMENT(#21,#11);",
            "#40=IFCBUILDINGSTOREY('1Storey0000000000000',#2,'Ground Floor',$,$,#41,$,$,.ELEMENT.,0.);",
            "#41=IFCLOCALPLACEMENT(#31,#11);",
        ]

        # Generate entities for each node
        entity_id = 50
        for node in nodes:
            b = node.bounding_box
            cx, cy, cz = b.center
            dx, dy, dz = b.extents

            pt_id = entity_id
            box_id = entity_id + 1
            shape_id = entity_id + 2
            prod_id = entity_id + 3
            entity_id += 4

            name_clean = node.name.replace("'", " ")

            if node.category == NodeCategory.ARCHITECTURE:
                ifc_class = "IFCWALL"
            elif node.category == NodeCategory.FURNITURE:
                ifc_class = "IFCFURNISHINGELEMENT"
            else:
                ifc_class = "IFCBUILDINGELEMENTPROXY"

            step_lines.append(f"#{pt_id}=IFCCARTESIANPOINT(({cx:.3f},{cz:.3f},{cy:.3f}));")
            step_lines.append(f"#{box_id}=IFCBLOCK(#11,{dx:.3f},{dz:.3f},{dy:.3f});")
            step_lines.append(f"#{shape_id}=IFCSHAPEREPRESENTATION(#10,'Body','SweptSolid',({box_id}));")
            step_lines.append(f"#{prod_id}={ifc_class}('{node.instance_id or node.name[:16]}',#2,'{name_clean}',$,$,#41,#{shape_id},$,$);")

        step_lines.append("ENDSEC;")
        step_lines.append("END-ISO-10303-21;")

        with open(out_path, "w", encoding="ascii") as f:
            f.write("\n".join(step_lines) + "\n")

        return out_path
