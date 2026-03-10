# Formal Proof: Energy-Based OOD Separation

**Theorem:**  
Let $x$ (in-distribution) and $x'$ (out-of-distribution). If $P_\theta(x) > P_\theta(x')$, then setting a threshold $\tau$ with $E_\theta(x) < \tau < E_\theta(x')$ allows correct separation.

**Proof:**  
By definition,
$$
E_\theta(x) = -\log P_\theta(x)
$$
$$
E_\theta(x') = -\log P_\theta(x')
$$
Since $P_\theta(x) > P_\theta(x')$, then $E_\theta(x) < E_\theta(x')$.  
Choosing $\tau$ such that $E_\theta(x) < \tau < E_\theta(x')$ means $x$ is classified as in-distribution ($E_\theta < \tau$), $x'$ as OOD ($E_\theta > \tau$).

$\Box$
