# TODO Review - Logging & Planning

## Current TODO Status

### 20+ TODOs Found in Code (Mostly in data_models.py)

#### Layer-Level TODOs (High Priority)
1. **All models need full `__post_init__` validation and serialization methods** (line 15)
   - Location: src/data_models.py:15
   - Impact: Data integrity, error prevention
   - Status: ⚠️ Not implemented

2. **Performance considerations for large batches documented at end of file** (line 16)
   - Location: src/data_models.py:16
   - Impact: Documentation only (performance notes)
   - Status: ⚠️ Not documented

#### BoundingBox Validations (Lines 181-183)
3. Ensure `x_min < x_max, y_min < y_max` (line 181)
4. Ensure coordinates are non-negative (line 182)
5. Ensure confidence is 0-1 if provided (line 183)
   - Impact: Prevent invalid geometries
   - Status: ⚠️ Not implemented in __post_init__

#### TableCell Validations (Lines 225-227)
6. Validate row/col indices are non-negative (line 225)
7. Validate colspan/rowspan >= 1 (line 226)
8. Validate confidence is 0-1 (line 227)
   - Impact: Prevent invalid table structures
   - Status: ⚠️ Not implemented in __post_init__

#### Table Structure TODOs (Lines 208-211, 255-257)
9. Handle merged cells properly (line 208)
10. Detect and extract background colors (line 209)
11. Preserve text styling (bold, italic) if detectable (line 210)
12. Add cell type classification (header, data, total, etc.) (line 211)
13. Handle ragged rows vs. fixed column counts (line 255)
14. Detect and preserve table structure with merged cells (line 256)
15. Extract table captions automatically (line 257)
   - Impact: Better table quality, structure preservation
   - Status: ⚠️ Design phase, not implemented

#### Method TODOs (Lines 247, 270)
16. Add methods for TableStructure utilities (line 247)
17. Compute num_rows and num_cols from cells (line 270)
   - Impact: Data utility, convenience
   - Status: ⚠️ Not implemented

#### Other TODOs
18. Implement classification logic (line 92)
19. Consider adding coordinate system normalization (line 144)
20. Implement validation (line 152)
21. Add methods for StructuralElement (line 166)
    - Impact: Data model completeness
    - Status: ⚠️ Varies by item

---

## Sprint Planning

### Sprint 3: List & Table Detection (Weeks 5-7)
**4 Tasks**, ~14 days
1. Implement marker-based list detection
2. Implement line-based table detection
3. Implement content-based table detection
4. Handle merged cells and irregular structures

**Dependencies**: 
- Requires data_models.py to be production-ready
- Table structures need validation
- May require cell type classification

**Blockers**:
- TODOs #9-12, #14-17 should be completed first

---

## Recommended Action Plan

### Phase 1: Fix Data Models (High Priority - ~1 week)
Focus on making data_models.py production-ready before Sprint 3 table detection:

1. **Add `__post_init__` validations to all models** ← Addresses TODO #1
   - BoundingBox: x_min < x_max, y_min < y_max, coords ≥ 0, confidence ∈ [0,1]
   - TableCell: row/col ≥ 0, colspan/rowspan ≥ 1, confidence ∈ [0,1]
   - StructuralElement: element_type valid, confidence ∈ [0,1]
   - All parent/child references valid

2. **Implement TableStructure utility methods** ← Addresses TODO #16-17
   - compute_grid_structure()
   - detect_merged_cells()
   - validate_table_integrity()

3. **Add table enhancement methods** ← Addresses TODOs #9-12, #14-15
   - detect_cell_types() - classify header vs data
   - extract_styling() - preserve bold/italic if detectable
   - extract_background_colors() - basic color detection
   - handle_ragged_rows() - normalize irregular tables

4. **Add StructuralElement utility methods** ← Addresses TODO #21
   - get_descendants() - tree traversal
   - flatten_hierarchy() - breadth-first traversal
   - compute_statistics() - element counts by type

### Phase 2: Implement Sprint 3 Tasks (~2-3 weeks)
Once data models are solid:
1. List detection (marker-based + spatial)
2. Table detection (line-based + content-based)
3. Handle edge cases (merged cells, irregular structures)

### Phase 3: Documentation (~1 week)
- Performance notes for large batches ← Addresses TODO #2
- Usage examples for all models
- Best practices guide

---

## Questions for User

**Which approach would you prefer?**

**Option A**: Jump to Sprint 3 (List & Table Detection)
- Skip data model validations for now
- Accept that tables/lists might have edge case issues
- Estimated: 2 weeks
- Risk: Data quality issues in complex documents

**Option B**: Fix Data Models First (Recommended)
- Implement all 20+ TODOs from data_models.py
- Make system production-ready
- Then do Sprint 3 with solid foundation
- Estimated: 3-4 weeks total
- Benefit: Robust, well-validated system

**Option C**: Hybrid Approach
- Fix critical validations only (#1, #3-5, #6-8)
- Defer table enhancements (#9-15)
- Do Sprint 3 with partial validation
- Estimated: 2.5 weeks
- Tradeoff: Some edge cases unhandled

---

## Logging Summary

### Where TODOs Are Logged
✅ **Inline Code Comments**: 20+ TODOs in src/data_models.py (lines 15-270)
✅ **Development Plan**: Sprint 3-9 tasks in OCR_DEVELOPMENT_PLAN.md
✅ **Documentation**: This file (TODO_REVIEW.md)

### Not Currently Logged
❌ Relative priority/urgency
❌ Estimated effort for each TODO
❌ Dependencies between tasks
❌ Risk assessment for each TODO

---

## Summary

| Item | Count | Logged? | Status |
|------|-------|---------|--------|
| Code-level TODOs | 20+ | ✅ Yes (data_models.py) | ⚠️ Not started |
| Sprint Tasks | 4-40 | ✅ Yes (dev plan) | ✅ S1-S2 done, S3+ pending |
| Data Model Validations | 8+ | ✅ Yes (comments) | ⚠️ Not implemented |
| Data Model Methods | 5+ | ✅ Yes (comments) | ⚠️ Not implemented |

**Recommendation**: Implement Phase 1 (data model fixes) before Sprint 3 to ensure system robustness.
