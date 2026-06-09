# Agent Instructions

- Output all mathematical formulas in LaTeX rendered in Codex. Do not overly use LaTeX for natural language; restrict it to math.
- Always commit and push after making an edit.
- For CryoSPARC Import Result Group workflows, keep imported result-group files under the CryoSPARC project directory. For project `CS-prestin`, use a path under `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/`, such as `imports/<name>/`. The `.csg` should reference the colocated `.cs` file by local filename, e.g. `filtered_particles.cs`, not an absolute `/nfs/.../filtered_particles.cs` path; otherwise CryoSPARC may prepend the job directory and fail import.
