## 2026-08-30: Phase 2 -- freeze mobile formats, design net-new schemas

Wrote `transforms.schema.json`, `trajectory_csv.schema.json`, `coverage_summary.schema.json` as
a transcription of `mobile/android`'s `DatasetFormat.kt` output (Phase 1 acceptance-passed, 92
green tests) -- no field renames/retypes, only `schema_version`/`$id`/`additionalProperties:
false` added on top. Designed `floorplan.schema.json`, `panorama_manifest.schema.json`,
`asset_manifest.schema.json` net-new per Phase_2.md §3-§5, including the constraints called out
as schema-enforced (equirect 2:1 ratio via enumerated `if`/`then` pairs, 25 MB splat
`compressed_bytes` ceiling, `asset_manifest.json`'s tier-conditional `if`/`then`/`else`, polygon
point-count, 2-vector wall endpoints). `validate.py` runs clean (`python
shared/schemas/validate.py` exits 0) and additionally checks two constraints Phase_2.md flags as
outside plain JSON Schema's reach: floorplan/panorama_manifest node-id cross-references, and
each wall's opening `offset_m + width_m <= wall length` (the latter needs the Euclidean distance
between `start`/`end`, which isn't expressible without a `$data` vocabulary extension --
deliberately kept out per Phase_2.md's own "more machinery than this needs" reasoning for the
sibling-reference case). `DatasetTest.kt` got two new tests tying `transforms.json`/
`coverage_summary.json` output to the same input values used in the shared fixtures (manual
sync point, not automated); `./gradlew :app:testDebugUnitTest --offline` is green.

Outcome: worked.

**Flag:** `floorplan.schema.json`'s coordinate frame (meters, gravity-aligned, origin at the
entry-door loop-closure anchor pose) is unvalidated against real backend gravity-alignment
output, since `backend/reconstruction/` doesn't exist yet (Phase 3). Per Phase_2.md §9,
reconcile via escalation rather than patching either side ad hoc if Phase 3's actual splat-volume
orientation turns out to disagree with this choice.
