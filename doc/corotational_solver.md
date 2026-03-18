# Geometrically Nonlinear Beam Solver: Co-rotational Formulation

This document describes the solution strategy for geometrically nonlinear beam networks
implemented in `beam_networks`. The element-level formulation follows Crisfield (1990)
[cited as C90 below] with several deliberate deviations noted where they occur.

---

## 1. Overview

The solver decomposes each beam element's motion into a **rigid-body part** (captured
by a co-rotating frame) and a **small deformational part** (captured by a linear local
stiffness). This separation allows geometrically exact treatment of large rotations and
displacements while keeping the local constitutive law linear — the core idea of the
*co-rotational* (CR) method.

A **Total Lagrangian** (TL) assembly strategy is used at the global level: all element
quantities are always computed relative to the original undeformed geometry, so the
natural element length $\ell_0$ is constant across all load steps. This eliminates the
path-dependence of the reference length that arises in Updated Lagrangian schemes and
makes results independent of the number of load steps.

The global equilibrium is solved by **load-stepped Newton–Raphson** iteration.

---

## 2. Co-rotating Frame

### 2.1 Two-dimensional case

For a 2D element connecting nodes $i$ and $j$, the reference chord direction is

$$\hat{\mathbf{e}}_0 = \frac{\mathbf{x}_j - \mathbf{x}_i}{\ell_0},$$

and the deformed chord direction is

$$\hat{\mathbf{e}}_1 = \frac{(\mathbf{x}_j + \mathbf{u}_j) - (\mathbf{x}_i + \mathbf{u}_i)}{\ell_n},$$

where $\ell_n$ is the current (deformed) chord length. The **chord rotation angle**
$\alpha$ is the angle from $\hat{\mathbf{e}}_0$ to $\hat{\mathbf{e}}_1$, computed
via the cross- and dot-products of the direction cosines:

$$\sin\alpha = c_0 s_n - s_0 c_n, \qquad \cos\alpha = c_0 c_n + s_0 s_n,$$
$$\alpha = \operatorname{arctan2}(\sin\alpha, \cos\alpha).$$

The `arctan2` call ensures a globally defined, sign-consistent result for all chord
angles including $|\alpha| = \pi$.

**Implementation:** `_rigid_body_rotation_2d` in `corotational.py:50–93`.

### 2.2 Three-dimensional case

For a 3D element the co-rotating frame is defined by three orthonormal vectors
$(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3)$.

**$\hat{\mathbf{e}}_1$** is the deformed chord direction (C90 eq. (23)):

$$\hat{\mathbf{e}}_1 = \frac{\mathbf{r}_j^n - \mathbf{r}_i^n}{\ell_n}.$$

**$\hat{\mathbf{e}}_2$** is defined by projecting a fixed **reference vector**
$\mathbf{v}_\mathrm{ref}$ perpendicular to $\hat{\mathbf{e}}_1$ and normalising
(Gram–Schmidt, Crisfield book eq. (17.57)):

$$\mathbf{p} = \mathbf{v}_\mathrm{ref}
  - \bigl(\mathbf{v}_\mathrm{ref}\cdot\hat{\mathbf{e}}_1\bigr)\hat{\mathbf{e}}_1,
\qquad
\hat{\mathbf{e}}_2 = \frac{\mathbf{p}}{|\mathbf{p}|}.$$

**$\hat{\mathbf{e}}_3 = \hat{\mathbf{e}}_1 \times \hat{\mathbf{e}}_2$.**

> **Deviation from C90:** Crisfield (1990) constructs $\hat{\mathbf{e}}_2$ and
> $\hat{\mathbf{e}}_3$ from an *average* nodal rotation matrix $\mathbf{R}_\mathrm{av}$
> (C90 eqs. (32)–(37)). The Gram–Schmidt approach used here is equivalent in accuracy
> but avoids tracking per-node rotation matrices, yields a simpler explicit B-matrix,
> and requires only one user-supplied reference vector per element. The trade-off is
> that the associated geometric stiffness contributions $K_{\sigma 2}\ldots K_{\sigma 5}$
> of C90 eq. (62) that arise from the variation of $\mathbf{R}_\mathrm{av}$ are absent.
> For the conservative problems considered (moment loading, tip rotations) these
> contributions vanish at convergence and do not affect accuracy.

**Implementation:** `_rigid_body_rotation_3d` in `corotational.py:96–154`.

---

## 3. Local Deformational DOFs

### 3.1 Two-dimensional case

The three local deformational DOFs per element are

$$\mathbf{u}_l = \begin{bmatrix} \Delta\ell \\ \theta_{0,l} \\ \theta_{1,l} \end{bmatrix}
= \begin{bmatrix} \ell_n - \ell_0 \\ \theta_0 - \alpha \\ \theta_1 - \alpha \end{bmatrix},$$

where $\theta_i$ are the global nodal rotation angles. Because the TL formulation
accumulates large rotations ($\theta > \pi$ is possible after many load steps), the
bending DOFs use a **wrap-safe angle-difference** formula instead of simple subtraction:

$$\theta_{i,l} = \operatorname{arctan2}\!\bigl(
  \sin\theta_i\cos\alpha - \cos\theta_i\sin\alpha,\;
  \cos\theta_i\cos\alpha + \sin\theta_i\sin\alpha
\bigr).$$

This is the trigonometric subtraction identity $\sin(\theta - \alpha)$,
$\cos(\theta - \alpha)$ and handles any accumulated rotation correctly.

**Implementation:** `_element_tangent_2d` in `corotational.py:571–587`.

### 3.2 Three-dimensional case

The seven local deformational DOFs per 3D element (C90 eqs. (25) and (28)) are

$$\mathbf{u}_l = \bigl[
  \Delta\ell,\;
  \theta_{x,0}^l,\; \theta_{y,0}^l,\; \theta_{z,0}^l,\;
  \theta_{x,1}^l,\; \theta_{y,1}^l,\; \theta_{z,1}^l
\bigr],$$

where the superscript $l$ denotes local-frame quantities and subscripts $0,1$ refer
to the two nodes.

**Axial DOF:** $\Delta\ell = \ell_n - \ell_0$.

**Torsion DOFs** (projections onto the deformed chord):
$\theta_{x,i}^l = \hat{\mathbf{e}}_1 \cdot \boldsymbol{\theta}_i$.

**Bending DOFs** require the chord rotation projected onto $\hat{\mathbf{e}}_2$ and
$\hat{\mathbf{e}}_3$. Define the chord rotation pseudo-vector

$$\mathbf{w} = \hat{\mathbf{e}}_0 \times \hat{\mathbf{e}}_1, \qquad
s = |\mathbf{w}| = \sin\varphi, \qquad
c = \hat{\mathbf{e}}_0 \cdot \hat{\mathbf{e}}_1 = \cos\varphi,$$

where $\varphi = \operatorname{arctan2}(s, c) \in [0,\pi]$ is the chord rotation angle.
The generalised chord rotation angles about $\hat{\mathbf{e}}_2$ and $\hat{\mathbf{e}}_3$
are

$$\alpha_{e_2} = \frac{\varphi}{s}\,(\mathbf{w}\cdot\hat{\mathbf{e}}_2), \qquad
\alpha_{e_3} = \frac{\varphi}{s}\,(\mathbf{w}\cdot\hat{\mathbf{e}}_3).$$

Because $|\mathbf{w}\cdot\hat{\mathbf{e}}_2| \le |\mathbf{w}| = s$, the product
$(\varphi/s)\cdot(\mathbf{w}\cdot\hat{\mathbf{e}}_2)$ is bounded by $\varphi \le \pi$
for all $s > 0$, even near $\varphi = \pi$. The guard is therefore simply

```python
# corotational.py:837–838
safe_s = np.where(s > 0, s, 1.)
scale  = np.where(s > 0, np.arctan2(s, c) / safe_s, 1.)
```

The fallback `scale = 1` is used only when $s = 0$ exactly (i.e. no rotation at all,
$\varphi = 0$), in which case $\mathbf{w}\cdot\hat{\mathbf{e}}_2 = 0$ as well and
$\alpha_{e_2} = 0$ is correct.

The bending DOFs then use the same wrap-safe formula as in 2D (helper `_wdiff`):

$$\theta_{y,i}^l = \operatorname{arctan2}\!\bigl(
  \sin(t_{e_2,i})\cos\alpha_{e_2} - \cos(t_{e_2,i})\sin\alpha_{e_2},\;
  \cos(t_{e_2,i})\cos\alpha_{e_2} + \sin(t_{e_2,i})\sin\alpha_{e_2}
\bigr),$$

where $t_{e_2,i} = \hat{\mathbf{e}}_2\cdot\boldsymbol{\theta}_i$, and analogously for
$\theta_{z,i}^l$ with $\hat{\mathbf{e}}_3$.

**Implementation:** `_exact_local_dofs_3d` in `corotational.py:779–872`.

---

## 4. Local Stiffness

### 4.1 Two-dimensional case

The $3\times 3$ local stiffness (Timoshenko beam) is

$$K_l = \begin{bmatrix}
EA/\ell_0 & 0 & 0 \\
0 & \psi & \xi \\
0 & \xi & \psi
\end{bmatrix},$$

with

$$\psi = \frac{(4+\Phi)\,EI}{\ell_0(1+\Phi)}, \qquad
\xi  = \frac{(2-\Phi)\,EI}{\ell_0(1+\Phi)}, \qquad
\Phi = \frac{12\,EI}{\kappa G A \,\ell_0^2}.$$

Setting $\kappa \to \infty$ (or $\Phi \to 0$) recovers Euler–Bernoulli.

**Implementation:** `_local_stiffness_2d` in `corotational.py:229–262`.

### 4.2 Three-dimensional case

The $7\times 7$ local stiffness decouples into axial, torsion, and two orthogonal
bending planes:

$$K_l = \mathrm{diag}\!\left(
  \tfrac{EA}{\ell_0},\;
  K_\mathrm{torsion},\;
  K_{\mathrm{bend},z},\;
  K_{\mathrm{bend},y}
\right),$$

where the torsion block is $\alpha_t[\mathbf{I}_{2\times2} - \mathbf{J}_{2\times2}]$
with $\alpha_t = GJ/\ell_0$, and each $2\times2$ bending block has diagonal $\psi$ and
off-diagonal $\xi$ entries computed from $EI_y$, $EI_z$ with their respective
Timoshenko parameters $\Phi_Y$, $\Phi_Z$.

**Implementation:** `_local_stiffness_3d` in `corotational.py:157–226`.

---

## 5. Kinematic (B) Matrix

The linearised kinematic operator maps virtual global DOFs $\delta\mathbf{p}$ to virtual
local DOFs $\delta\mathbf{p}_l$:

$$\delta\mathbf{p}_l = \mathbf{B}\,\delta\mathbf{p}.$$

### 5.1 Two-dimensional case

The $3\times 6$ B-matrix consists of:

- **Row 0 (axial):** $\mathbf{g} = [-c,\,-s,\,0,\,c,\,s,\,0]$, i.e. the deformed
  chord direction (C90 eq. (57)).
- **Rows 1–2 (bending):** $\pm s/\ell_n$ for translational DOFs and $1$ for the nodal
  rotation DOFs, implementing $\delta\theta_l = \delta\theta - \delta\alpha$ in the
  deformed configuration.

**Implementation:** `_b_matrix_2d` in `corotational.py:265–293`.

### 5.2 Three-dimensional case

The $7\times 12$ B-matrix extends the 2D logic:

- **Row 0 (axial):** $[-\hat{\mathbf{e}}_1,\,\mathbf{0},\,+\hat{\mathbf{e}}_1,\,\mathbf{0}]$.
- **Rows 1, 4 (torsion):** direct projection onto $\hat{\mathbf{e}}_1$.
- **Rows 2, 5 (bending $\theta_y$):** translation entries $\mp\hat{\mathbf{e}}_3/\ell_n$
  (moving a node in the $\hat{\mathbf{e}}_3$ direction rotates the chord about
  $\hat{\mathbf{e}}_2$), rotation entry $\hat{\mathbf{e}}_2$.
- **Rows 3, 6 (bending $\theta_z$):** translation entries $\pm\hat{\mathbf{e}}_2/\ell_n$
  (moving in $\hat{\mathbf{e}}_2$ rotates about $-\hat{\mathbf{e}}_3$), rotation entry
  $\hat{\mathbf{e}}_3$.

> **Deviation from C90:** Because the frame is defined by Gram–Schmidt projection
> rather than the average nodal rotation matrix $\mathbf{R}_\mathrm{av}$, the bending
> rows take a simpler form than C90 eqs. (54)–(55). The chord-rotation contribution
> to $\delta\theta_{y,l}$ is $\pm\hat{\mathbf{e}}_3/\ell_n$ rather than the paper's
> $\mathbf{F}$ matrix. The B-matrix is the linearisation of
> $\theta_{y,l} \approx \hat{\mathbf{e}}_2\cdot\boldsymbol{\theta}_i - \alpha_{e_2}$
> and correctly captures the translation-induced chord rotation, but omits the term
> $\mathrm{d}\alpha_{e_2}/\mathrm{d}\mathbf{u}$. This term is large near chord angle
> $\varphi = \pi$ (see Section 7) but vanishes at convergence for conservative loads.

**Implementation:** `_b_matrix_3d` in `corotational.py:296–369`.

---

## 6. Internal Forces and Tangent Stiffness

### 6.1 Internal forces

Local element forces follow from the linear constitutive relation in the co-rotating
frame (C90 eqs. (27) and (30)):

$$\mathbf{f}_l = K_l\,\mathbf{u}_l.$$

Global element internal forces are obtained by virtual work (C90 eq. (58)):

$$\mathbf{f}_g = \mathbf{B}^\top \mathbf{f}_l.$$

**Implementation:** `_element_tangent_2d/3d` in `corotational.py`.

### 6.2 Tangent stiffness

The global element tangent stiffness is the sum of material and geometric contributions
(C90 eq. (59)):

$$K_t = K_m + K_\sigma, \qquad K_m = \mathbf{B}^\top K_l \mathbf{B}.$$

The **geometric stiffness** $K_\sigma$ arises from $\mathbf{f}_l^\top\,\partial\mathbf{B}/\partial\mathbf{p}$, retaining only the chord-direction terms of B:

$$K_\sigma = \frac{N}{\ell_n}\!\left(\mathbf{E}_2\otimes\mathbf{E}_2
  + \mathbf{E}_3\otimes\mathbf{E}_3\right)
+ \frac{M_{z,0}+M_{z,1}}{\ell_n^2}\!\left(\mathbf{E}_1\otimes\mathbf{E}_2
  + \mathbf{E}_2\otimes\mathbf{E}_1\right)
- \frac{M_{y,0}+M_{y,1}}{\ell_n^2}\!\left(\mathbf{E}_1\otimes\mathbf{E}_3
  + \mathbf{E}_3\otimes\mathbf{E}_1\right),$$

where $\mathbf{E}_k = [-\hat{\mathbf{e}}_k,\,\mathbf{0},\,+\hat{\mathbf{e}}_k,\,\mathbf{0}]$
are 12-component translation-only vectors, $N$ is the axial force, and $M_{y/z,0/1}$ are
the nodal bending moments. The axial term corresponds to C90 eq. (64) ($K_{11} = NA$).
The 2D geometric stiffness has the same structure; see Crisfield, *Non-linear FEA*
Vol. 1, eqs. (3.28)–(3.30).

> **Deviation from C90:** The terms $K_{\sigma 2}\ldots K_{\sigma 5}$ of C90 eq. (62),
> which arise from the variation of the average rotation matrix $\mathbf{R}_\mathrm{av}$,
> are absent because the Gram–Schmidt frame has no nodal-rotation dependence in e2/e3.

**Implementation:** `_geometric_stiffness_3d` in `corotational.py:401–483`,
`_element_tangent_2d` in `corotational.py:602–616`.

---

## 7. Global Assembly and Boundary Conditions

After computing element contributions, the global stiffness $K$ and internal force
vector $\mathbf{F}_\mathrm{int}$ are assembled by standard scatter operations.

The system is partitioned into free DOFs $\mathcal{F}$ (unconstrained) and Dirichlet
DOFs $\mathcal{D}$ (prescribed). The Newton–Raphson equation is solved only for the
free partition:

$$K_{\mathcal{F}\mathcal{F}}\,\Delta\mathbf{u}_\mathcal{F}
= -\left(\mathbf{F}_\mathrm{int} - \mathbf{F}_\mathrm{target}\right)_\mathcal{F}.$$

Two assembly backends are available: dense (`numpy.linalg.solve`) and BSR sparse
(`scipy.sparse.linalg.spsolve`).

**Implementation:** `assemble_nonlinear_system_{2d,3d}`, `partition_stiffness`
in `corotational.py` and `fem/partitioning.py`.

---

## 8. Load-Stepped Newton–Raphson Solver

**Implementation:** `solve_nonlinear` in `solvers/nonlinear.py`.

### 8.1 Load increments

The total load is divided into $n_\mathrm{steps}$ equal increments. At step $k$:

- **Neumann (applied force):** cumulative target load
  $\mathbf{F}_\mathrm{target}^{(k)} = k\,\Delta\mathbf{f}$.
- **Dirichlet (prescribed displacement):** cumulative prescribed value
  $\bar{u}_i^{(k)} = k\,\bar{u}_i^\mathrm{total} / n_\mathrm{steps}$.

The Dirichlet DOFs are set *before* Newton iteration:

```python
# nonlinear.py:181
sol_total[dof_D] = (step + 1) * d_D_step[dof_D]
```

### 8.2 Newton–Raphson iteration

Within each load step, Newton–Raphson iterations update the free DOFs:

$$\mathbf{u}_\mathcal{F}^{(i+1)} = \mathbf{u}_\mathcal{F}^{(i)}
+ K_{\mathcal{F}\mathcal{F}}^{-1}\!\left(
  \mathbf{F}_\mathrm{target} - \mathbf{F}_\mathrm{int}\!\left(\mathbf{u}^{(i)}\right)
\right)_\mathcal{F}.$$

Convergence is declared when the Euclidean norm of the correction falls below the
tolerance:

$$\|\Delta\mathbf{u}_\mathcal{F}\| < \varepsilon \quad (\varepsilon = 10^{-9}
\text{ by default}).$$

The Total Lagrangian reference is the original undeformed mesh, so $\ell_0$ and all
reference quantities are constant across load steps.

```python
# nonlinear.py:193–208
for it in range(max_iter):
    K, F_int = _assemble(sol_total)
    du_F = solve(K_FF, -(F_int - f_target)[dof_F])
    sol_total[dof_F] += du_F
    if norm(du_F) < tol:
        break
```

### 8.3 Tangent predictor (3D only)

At the start of each load step, the free DOFs are initialised by a **linear tangent
predictor**: the increment $\Delta\mathbf{u}_\mathcal{F}^{(k-1)}$ from the previous
converged step is added to the current free DOF values:

$$\mathbf{u}_\mathcal{F}^{(k),\mathrm{pred}}
= \mathbf{u}_\mathcal{F}^{(k-1)}
+ \underbrace{\left(\mathbf{u}_\mathcal{F}^{(k-1)} - \mathbf{u}_\mathcal{F}^{(k-2)}\right)}_{\Delta\mathbf{u}_\mathcal{F}^{(k-1)}}.$$

```python
# nonlinear.py:184–187
if ndim == 3:
    curr_sol_F = sol_total[dof_F].copy()
    sol_total[dof_F] += curr_sol_F - prev_sol_F
    prev_sol_F = curr_sol_F
```

**Motivation:** The 3D co-rotational formulation has a kinematic singularity at chord
angle $\varphi = \pi$. When an element chord reaches exactly $\pi$ in the converged
solution of step $k$, the B-matrix used to build the tangent stiffness omits the term
$\mathrm{d}\alpha_{e_2}/\mathrm{d}\mathbf{u}$, which diverges at $\varphi = \pi$
(since $\mathrm{d}(\varphi/s)/\mathrm{d}s \to -\infty$). As a result, Newton's search
direction for step $k+1$ is inaccurate near the singularity, and for small Dirichlet
increments Newton can converge to a spurious equilibrium on the wrong solution branch.
The linear predictor ensures that Newton starts *past* the $\pi$ singularity (on the
correct side), where the B-matrix approximation is again adequate. The predictor does
not affect accuracy at convergence.

The predictor is applied for 3D only. In 2D the bending DOFs are computed from
$\operatorname{arctan2}(\Delta y, \Delta x)$, which is globally defined for all chord
angles including $\varphi = \pi$, so no such singularity exists and the predictor is
not needed (and would cause overshooting for large step sizes).

---

## 9. Summary of Deviations from Crisfield (1990)

| Aspect | C90 | This implementation |
|--------|-----|---------------------|
| Local transverse axes $\hat{\mathbf{e}}_2$, $\hat{\mathbf{e}}_3$ | Average nodal rotation matrix $\mathbf{R}_\mathrm{av}$, eqs. (32)–(37) | Gram–Schmidt projection of a fixed reference vector |
| Geometric stiffness | Full $K_\sigma$ including $K_{\sigma 2}\ldots K_{\sigma 5}$ from $\delta\mathbf{R}_\mathrm{av}$, eq. (62) | Chord-direction terms only (N, $M_y$, $M_z$) |
| Bending DOF formula | $2\sin\theta$ relations, eq. (28) | Exact wrap-safe $\operatorname{arctan2}$ formula |
| B-matrix bending rows | $\mathbf{F}$ matrix involving $\mathbf{R}_\mathrm{av}$, eqs. (54)–(55) | Simplified $\pm\hat{\mathbf{e}}_2/\ell_n$, $\pm\hat{\mathbf{e}}_3/\ell_n$ |
| Reference configuration | Updated Lagrangian (implicit) | Total Lagrangian (explicit) |
| Scale guard at $\varphi\to\pi$ | Not addressed | Threshold `s > 0`; $\varphi/s$ evaluated at float-precision $s$ |
| Step predictor | Not applicable | Linear tangent predictor for 3D Dirichlet steps |

---

## References

Crisfield, M. A. (1990). A consistent co-rotational formulation for non-linear,
three-dimensional, beam-elements. *Computer Methods in Applied Mechanics and
Engineering*, **81**(2), 131–150.

Crisfield, M. A. (1991). *Non-linear Finite Element Analysis of Solids and Structures,
Volume 1*. Wiley.
