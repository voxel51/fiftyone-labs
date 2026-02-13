### Neeraja's planning

For the two operators in `__init__.py`, I wanna put together a Panel that lets people use the operators interactively.

Example workflow
1. A dataset/view is loaded. I open a "Label Propagation" Panel.
2. I run the AssignExemplarFrames. It populates all samples with exemplar_frame_field.is_exemplar (bool), and exemplar_frame_field.exemplar_assignment=[list of IDs].
3. All possible ids in the exemplar_assignments of any sample --> my "exemplars". Each of them has a "propagation view" which is the set of all samples for which selected_exemplar is in the exemplar_assignment. This view is ordered by whatever sort field I used in AssignExemplarFrames.
4. Would be best if these views are arranges as tabs on the panel and I can just click on one.
5. Alternatively, or in addition, I can now click on a sample and say something like "open propagation view".. If the sample is an exemplar, selected_exemplar = sample.id. Else, selected_exemplar = exemplar_assignment[0] of the ID.
6. When I open a Propagation View, the selected view changes back in the "Samples" panel. The exemplar_frame_field.is_exemplar box is checked. Meanwhile, the "Label Propagation" Panel shows the selected exemplar for this view.
7. I can then click on the exemplar sample (or actually any sample) and annotate in the sample modal (assumed that annotation is enabled).
8. Then, while still on the Propagation View, I can call the propagate operator. After propagation, I can choose to edit my annotations (assume this is possible -- no need to do anything here).
9. Keep switching tabs, and continue!


Questions:
- [ ] Panel in Python sufficient? Or JS necessary?
- [ ] 


---

### Execution Planning

**Key Corrections:**
- Configuration inputs (`exemplar_frame_field` and `sort_field`) are always shown at the top
- `AssignExemplarFrames` is optional - if field exists and is fully populated, show exemplars; if partially populated, don't pre-populate (user needs to run operator anyway)
- Both operators use the same `sort_field` from the top configuration inputs

**Updated Flow:**
1. Panel always shows configuration inputs at top
2. Check if exemplar field exists and is fully populated
3. If yes: show exemplar dropdown with discovered exemplars
4. If no/partial: show message to run AssignExemplarFrames
5. AssignExemplarFrames UI always available (allows re-running)
6. PropagateLabels uses sort_field from top config inputs

See detailed plan in `.cursor/plans/label_propagation_panel_37403f5e.plan.md`

---