#@ -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Cours ENPC - Pratique du calcul scientifique

# ## Examen final
#
# - Ce notebook est à soumettre sur <a href="https://educnet.enpc.fr/mod/assign/view.php?id=58482">Educnet</a> avant 16h30.
#
# - L’examen comporte trois exercices indépendants. Dans chaque exercice les
#   cellules peuvent éventuellement dependre des cellules précèdentes.
#
# - Afin de faciliter l'évaluation de votre code,
#   ne pas changer les signatures des fonctions à implémenter.
#
# - La cellulle ci-dessous importe les packages utilisés dans ce notebook et
#   définit une macro qui a pour but de faciliter les tests unitaires figurant
#   dans le sujet.

# +
using Polynomials
using Plots
using LaTeXStrings
using LinearAlgebra

macro mark(bool_expr)
    return :(print($bool_expr ? "✔️" : "❌"))
end
# -

# ### 1. Intégration numérique

# 1. Écrire une fonction `composite_trapezoidal(u, a, b, n)` permettant d'approximer l'intégrale
#    $$
#    I := \int_a^b u(x) \, \mathrm d x
#    $$
#    par la méthode trapézoidale composite avec `n` de points équidistants $a = x_1 < x_2 < \dots < x_{n-1} < x_n = b$.
#    On supposera que $n \geq 2$.

# +
function composite_trapezoidal(u, a, b, n)

end

@mark composite_trapezoidal(x -> 5, 1, 2, 100) ≈ 5
@mark composite_trapezoidal(x -> x, 1, 2, 100) ≈ 3/2
@mark composite_trapezoidal(x -> x, 1, 2, 2) ≈ 3/2
@mark composite_trapezoidal(x -> x^2, 0, 1, 2) ≈ 1/2
@mark composite_trapezoidal(x -> x^2, 1, 2, 2) ≈ 5/2
# -

# 2. Écrire une fonction `composite_simpson(u, a, b, n)` permettant d'approximer l'intégrale $I$ par la méthode de Simpson composite
#    basée sur des évaluations de `u` à un nombre **impair** `n` de points équidistants tels que $a = x_1 < x_2 < \dots < x_{n-1} < x_n = b$.
#    On supposera que $n$ est impair et $n \geq 3$.
#
#    > **Remarque**: `n` est ici le nombre de points auxquels la fonction `u` est évaluée,
#    > et pas un nombre d'intervalles où la règle de Simpson est appliquée localement.

# +
function composite_simpson(u, a, b, n)
    @assert n % 2 == 1 "`n` must be impair"

end

@mark composite_simpson(x -> 1  , 1, 2, 101) ≈ 1
@mark composite_simpson(x -> x  , 1, 2, 101) ≈ 3/2
@mark composite_simpson(x -> x^2, 1, 2, 101) ≈ 7/3
@mark composite_simpson(x -> x^3, 1, 2, 101) ≈ 15/4
@mark composite_simpson(x -> x  , 0, 1, 3) ≈ 1/2
@mark composite_simpson(x -> x^2, 0, 1, 3) ≈ 1/3
@mark composite_simpson(x -> x^3, 0, 1, 3) ≈ 1/4
# -

# 3. Écrire une fonction `calculate_sum(N)` qui permet de calculer la somme
#    $$
#    S(n) := \sum_{n = 1}^{N} n^{-n}.
#    $$
#    Afficher la valeur de $S(N)$ pour $n$ égal à 5, 10, 15, et 20.

# +
function calculate_sum(N)

end

@mark abs(calculate_sum(20) - 1.2912859970626636) < 1e-6
@mark abs(calculate_sum(20) - 1.2912859970626636) < 1e-9
@mark abs(calculate_sum(20) - 1.2912859970626636) < 1e-12
# -

# 4. On peut montrer que
#    $$
#    \int_0^1 x^{-x} \, \mathrm d x = \sum_{n=1}^{\infty} n^{-n}.
#    $$
#    Illustrer l'erreur des méthodes composites définies ci-dessus en fonction de `n`,
#    sur un même graphe.
#    Utiliser $S(20)$ comme valeur de référence pour l'intégrale,
#    et employer l'échelle logarithmique pour les deux axes du graphe.
#
#    > **Remarque**: La fonction à intégrer dans cet exercice est continue,
#    > mais sa dérivée diverge en $x = 0$.
#    > Ne vous inquiétez donc pas si le taux de convergence que vous observez ne correspond pas au taux théorique.

# +
# -

# 5. (**Bonus**). Estimer, en approximant à l'aide de la fonction `fit` le logarithme de l'erreur par une fonction affine du logarithme du pas d'intégration,
# l'ordre de convergence de la méthode composite de Simpson pour le calcul de l'intégrale dans la question précédente.

# +
# -

# ### 2. Résolution d'un système linéaire

# L'objectif de cet exercice est de proposer un algorithme permettant de réaliser la décomposition LU d'une matrice réelle $\mathsf{A}\in\mathbb{R}^{n×n}$,
# **non pas par élimination gaussienne** mais par identification des entrées de $\mathsf A$ avec celles de $\mathsf L \mathsf U$.
# Il s'agit de trouver un matrice triangulaire inférieure $\mathsf L$ formée de 1 sur la diagonale
# et une matrice triangulaire supérieure $\mathsf U$ telles que :
# <a id="LU"></a>
# $$
# \tag{LU}
# \mathsf{A}=\mathsf{L}\mathsf{U}
# $$
#
# 1. Écrire une fonction `my_lu(A)` qui prend comme argument une matrice `A` et qui renvoie les matrices `L` et `U`.
#    Pour calculer ces matrices, s'appuyer sur une identification successive des éléments des deux membres de <a href="#LU">(LU)</a>,
#    ligne par ligne de haut en bas, et de gauche à droite au sein de chaque ligne.
#
#    > **Indication**: lorsqu'on suit l'ordre conseillé,
#    > la comparaison de l'élément $(i, j)$ fournit une équation pour $\ell_{ij}$ si $j < i$,
#    > et une équation pour $u_{ij}$ si $j \geq i$.
#    > Notons qu'il est possible de parcourir les éléments dans d'autres ordres.

# +
function my_lu(A)

end

@mark my_lu(diagm([1; 2; 3])) == (diagm([1; 1; 1]), diagm([1; 2; 3]))
@mark my_lu([2 -1 0; -1 2 -1; 0 -1 2])[1] ≈ [1 0 0; -1/2 1 0; 0 -2/3 1]
@mark my_lu([2 -1 0; -1 2 -1; 0 -1 2])[2] ≈ [2 -1 0; 0 3/2 -1; 0 0 4/3]
@mark begin C = [1 2 3 4; 4 3 2 1; 1 2 1 2; 1 5 4 1]; my_lu(C)[1] ≈ lu(C, NoPivot()).L end
@mark begin C = [1 2 3 4; 4 3 2 1; 1 2 1 2; 1 5 4 1]; my_lu(C)[2] ≈ lu(C, NoPivot()).U end
@mark begin C = randn(100, 100); my_lu(C)[1] ≈ lu(C, NoPivot()).L end
@mark begin C = randn(100, 100); my_lu(C)[2] ≈ lu(C, NoPivot()).U end
# -

# 2. On suppose maintenant que la matrice réelle définie positive `A` est à largeur de bande `b` supposée beaucoup plus petite que `n`.
#    Réécrire la fonction de décomposition LU en exploitant la largeur de bande.

# +
function my_banded_lu(A, b)

end

@mark begin C = [1 2 3 4; 4 3 2 1; 1 2 1 2; 1 5 4 1]; my_lu(C)[1] ≈ lu(C, NoPivot()).L end
@mark begin C = [1 2 3 4; 4 3 2 1; 1 2 1 2; 1 5 4 1]; my_lu(C)[2] ≈ lu(C, NoPivot()).U end
@mark begin C = randn(100, 100); my_banded_lu(C, 100)[1] ≈ lu(C, NoPivot()).L end
@mark begin C = randn(100, 100); my_banded_lu(C, 100)[2] ≈ lu(C, NoPivot()).U end
# -

# 3. Construire une fonction `generate_banded(n, b)` permettant de générer une matrice carrée aléatoire de taille `n` à largeur de bande donnée `b`.

# +
function generate_banded(n, b)

end

@mark generate_banded(10, 2)[1, 5] == 0
@mark generate_banded(10, 2)[2, 5] == 0
@mark generate_banded(10, 2)[3, 5] != 0
@mark generate_banded(10, 2)[4, 5] != 0
@mark generate_banded(10, 2)[5, 5] != 0
@mark generate_banded(10, 2)[6, 5] != 0
@mark generate_banded(10, 2)[7, 5] != 0
@mark generate_banded(10, 2)[8, 5] == 0
@mark generate_banded(10, 2)[9, 5] == 0
# -

# 4. En utilisant `generate_banded`, tester votre implémentation de `my_banded_lu`,
#    pour `n = 100` et des valeurs de `b` égales à 2, 3 et 10.
#    Utiliser la fonction `lu` de la bibliothèque `LinearAlgebra`,
#    avec l'argument `NoPivot()`, comme fonction de référence.
#    Vous pouvez également utiliser la macro `@mark` pour vos tests.

# +
# -

# ### 3. Résolution d'une équation différentielle
#
# Cet exercice vise à calculer la trajectoire d'une petite fusée
# et à dimensionner son chargement en carburant en vue d'atteindre un certain objectif.
# On fera plusieurs hypothèses simplificatrices :
#
# - On néglige la rotation de la terre.
#
# - On néglige la variation de l'accélération de gravité $g$ et de la densité de l'air $\rho$ avec l'altitude.
#
# - On suppose que le coefficient de traînée $C^d$ est indépendant du nombre de Reynolds de l'écoulement.
#
# - La mouvement de la fusée est confiné à l'axe verticale. Son altitude et sa vitesse au lancement sont $z = 0$ et $v = 0$.
#
# - La fusée est approximée par un cylindre de rayon $r$ (sa hauteur n'a pas d'importance pour cet exercice).
#
# - Le carburant est éjecté à une vitesse constante par rapport à la fusée, notée $V_e$.
#
# - Le carburant est consommé à un taux $\beta(\mu)$,
#   dépendant uniquement de la masse $\mu$ de carburant restant.
#
# On note $z(t)$ l'altitude de la fusée, $v(t)$ sa vitesse et $\mu(t)$ la masse de carburant restant.
# Sous les hypothèses que nous avons faites,
# le mouvement de la fusée peut-être modélisé par le système d'équations différentielles suivant:
# $$
# \tag{Rocket}
# \left\{
# \begin{aligned}
# z'(t) &= v(t), \\
# m(t) v'(t) &= \beta\bigl(\mu(t)\bigr) V_e - F^d\bigl(v(t)\bigr) - m(t) g, \\
# \mu'(t) &= - \beta\bigl(\mu(t)\bigr).
# \end{aligned}
# \right.
# \qquad
# \left\{
# \begin{aligned}
# z(0) &= 0, \\
# v(0) &= 0, \\
# \mu(0) &= \mu_0.
# \end{aligned}
# \right.
# $$
# <a id="Rocket"></a>
# Ici, $\mu_0$ est la masse de carburant au lancement,
# et $m(t) = m_r + \mu(t)$ est la masse totale de la fusée,
# qui comporte une partie fixe $m_r$, correspondant à la structure et à la cargaison,
# et une partie variable $\mu(t)$ correspondant au carburant.
# L'expression de la force de trainée $F^d$ est donnée dans la cellule ci-dessous.
# On fera varier au cours de l'exercice les paramètres $\mu_0$ et $C^d$,
# et ceux-ci ne sont donc pas définis ci-dessous.

# +

# Air density at 0 ⁰C [kg/m³]
const ρ = 1.293

# Gravity acceleration [m/s²]
const g = 9.81

# Radius of the rocket [m]
const r = .1

# Cross-sectional area [m²]
const A = π*r^2

# Effective exhaust velocity [m/s]
const Vₑ = 1000

# Mass of the rocket without fuel [kg]
const mᵣ = 5

# Burn rate in the limit where μ → ∞ [kg/s]
const β₊ = 1

# Burn rate function [kg/s]
β(μ) = μ > 0. ? β₊ * tanh(μ) : 0.

# Drag force depending on v
Fᵈ(v, Cᵈ) = 1/2 * ρ*A*Cᵈ*v^2
# -

# 1. Le problème <a href="#Rocket">(Rocket)</a> décrit peut être réécrit comme un problème aux valeurs initiales du premier ordre pour le vecteur $\mathbf X := (z, v, \mu)^T$ de la forme suivante:
#    $$
#    \mathbf X'(t) = \mathbf f\bigl(t, \mathbf X(t), C^d \bigr), \qquad \mathbf X(0) = \mathbf X_0.
#    \tag{1st-order}
#    $$
#    <a id="1st-order"></a>
#    Écrire la fonction $f$ sous forme d'une fonction Julia `f(t, X, Cᵈ)`.


# +
function f(t, X, Cᵈ)

end

@mark f(0, [0, 0, 0], 0) == [0; -g; 0]
@mark f(1, [0, 0, 0], 0) == [0; -g; 0]
@mark f(1, [0, 0, 0], 5) == [0; -g; 0.]
@mark f(0, [0, 0, 1], 0) ≈ [0.; Vₑ * β(1) / (mᵣ + 1)  - g; - β(1)]
@mark f(0, [0, 100, 5], 0) ≈ [100.; Vₑ * β(5) / (mᵣ + 5)  - g; - β(5)]
@mark f(1, [5, 5, 5], 1) ≈ [5; Vₑ * β(5) / (mᵣ + 5) - Fᵈ(5, 1) / (mᵣ + 5) - g; - β(5)]
# -

# 2. Écrire une fonction `rkx(tₙ, Xₙ, f, Δ)` implémentant un pas de temps de taille $\Delta$ de la méthode suivante de Runge-Kutta suivante pour une équation différentielle générique de la forme $X' = h(t, X)$:
#    $$
#       \mathbf X_{n+1} = \mathbf X_n + \frac{\Delta}{9}\left(2\mathbf k_1 + 3\mathbf k_2 + 4\mathbf k_3 \right),
#    $$
#    où
#    \begin{align*}
#    \mathbf k_1 &= \ h(t_n, \mathbf X_n), \\
#    \mathbf k_2 &= \ h\!\left(t_n + \frac{\Delta}{2}, \mathbf X_n + \frac{\Delta}{2} \mathbf k_1\right), \\
#    \mathbf k_3 &= \ h\!\left(t_n + \frac{3\Delta}{4}, \mathbf X_n + \frac{3\Delta}{4} \mathbf k_2\right).
#    \end{align*}
#    La fonction devra renvoyer $\mathbf X_{n+1}$.

# +
function rkx(tₙ, Xₙ, h, Δ)

end

@mark rkx(0., [0.], (t, X) -> [1.], 1.) ≈ [1]
@mark rkx(1., [0.], (t, X) -> [t], 1.)  ≈ [3/2]
@mark rkx(0., [0.; 0.; 0.], (t, X) -> [1, t, t^2], 1.) ≈ [1; 1/2; 1/3]
# -

# 3. Écrire une fonction `solve_ode(Δ, Cᵈ, μ₀)` permettant de résoudre <a href="#1st-order">(1st-order)</a> pour les paramètres $C^d$ et $μ_0$ donnés en arguments,
#    en utilisant la méthode de Runge-Kutta donnée ci-dessus avec pas de temps fixe `Δ`.
#    Votre fonction devra renvoyer un vecteur de temps `ts` et un vecteur de vecteurs `Xs` contenant la solution à ces temps.
#    On calculera la trajectoire de la fusée jusqu'à la fin de son ascension.
#    Plus précisément, on demande d'interrompre l'intégration numérique dès que la valeur de $v$ sera devenue strictement négative;
#    il faudra donc que `Xs[end-1][2]` soit positif et `Xs[end][2]` soit strictement négatif.

# +
function solve_ode(Δ, Cᵈ, μ₀)

end

@mark solve_ode(.01, 0, 5) |> length == 2
@mark solve_ode(.01, 0, 5)[2][end-1][2] ≥ 0
@mark solve_ode(.01, 0, 5)[2][end][2] ≤ 0
@mark solve_ode(.01, 0, 5)[1][1:10] ≈ 0:.01:.09
# -

# 4. Écrire une fonction `plot_altitude(Δ, Cᵈ, μ₀)` permettant d'illustrer sur un même graphe l'altitude de la fusée en fonction du temps,
# pour **une** valeur de `Cᵈ` donnée et **plusieurs** valeurs de $\mu_0$ dans le vecteur `μs`.

# +
function plot_altitude(Δ, Cᵈ, μs)
    p = plot(title="Altitude of the rocket")

    return p
end

Δ, Cᵈ, μs = .01, .75, [1; 2; 3; 4; 5]
plot_altitude(Δ, Cᵈ, μs)
# -

# 5. Écrire une fonction `plot_velocity(Δ, Cᵈ, μ₀)` permettant d'illustrer sur un même graphe la vitesse de la fusée en fonction du temps,
# pour **une** valeur de $Cᵈ$ donnée et **plusieurs** valeurs de $\mu_0$ dans le vecteur `μs`.

# +
function plot_velocity(Δ, Cᵈ, μs)
    p = plot(title="Velocity of the rocket")

    return p
end

Δ, Cᵈ, μ₀ = .01, .75, [1; 2; 3; 4; 5]
plot_velocity(Δ, Cᵈ, μ₀)
# -

# 6. On suppose ici que $C^d = 0$.
#    Dans ce cas, une équation fondamentale de l'astronautique connue sous le nom d'équation de Tsiolkovski
#    donne la vitesse de la fusée en fonction de sa masse:
#    $$
#    v(t) = V_e \log\left(\frac{m(0)}{m(t)}\right) - g t.
#    $$
#    Il suffit donc de connaître la masse de la fusée à un instant donné pour en connaître sa vitesse.
#    Or, la troisième équation du système <a href="#Rocket">(Rocket)</a> peut être résolue analytiquement:
#    $$
#    \mu(t) = \sinh^{-1} \Bigl( \exp\bigl(  \log(\sinh(\mu_0)) - t \bigr) \Bigr).
#    $$
#    Il est donc possible d'obenir une expression analytique de la fonction vitesse $v(t)$.
#    Écrire une fonction `error_velocity(Δ, μ₀)` qui renvoit l'erreur en norme maximum entre cette fonction et la composante vitesse de la solution numérique,
#    définie comme
#    $$
#    e(\Delta) := \max_{i} \bigl\lvert v(t_i) - \widehat v_i \bigr\rvert,
#    $$
#    où $\widehat v_i$ est l'approximation numérique de la vitesse au temps $i \Delta$.

# +
μ_exact(t, μ₀) = asinh(exp(log(sinh(μ₀)) - t))
v_exact(t, μ₀) = Vₑ * log((mᵣ + μ₀) / (mᵣ + μ_exact(t, μ₀))) - g*t

function error_velocity(Δ, μ₀)

end

@mark error_velocity(.1, 5) < 1e-2
@mark error_velocity(.1, 5) > 1e-3
@mark error_velocity(.01, 5) < 1e-5
@mark error_velocity(.01, 5) > 1e-6
@mark error_velocity(.001, 5) < 1e-8
@mark error_velocity(.001, 5) > 1e-9
# -

# 7. Tracer un graphique de l'erreur en fonction de $Δ$ pour les valeurs de $Δ$ données ci-dessous,
#    et pour $μ₀ = 5$.
#    Utiliser l'échelle logarithmique pour les deux axes.

# +
Δs, μ₀ = 2.0 .^ (-10:-2), 5
# -

# 8. Écrire une fonction `max_altitude(Δ, Cᵈ, μ₀)` renvoyant l'altitude maximale de la fusée pour une quantité de carburant $μ₀$ donnée,
#    approximée en utilisant un pas de temps `Δ` pour les paramètres `Cᵈ` et `μ₀`.

# +
function max_altitude(Δ, Cᵈ, μ₀)

end

@mark max_altitude(.0001, .75, 1) |> floor == 452
@mark max_altitude(.0001, .75, 2) |> floor == 760
@mark max_altitude(.0001, .75, 3) |> floor == 997
# -

# 9. Faire un plot de l'altitude atteinte par la fusée pour $Cᵈ = .75$ et les valeurs de $\mu_0$ données ci-dessous,
#    estimée avec $\Delta = .01$ [s].
#    Estimer graphiquement la valeur minimale de $\mu_0$ permettant d'atteindre des altitudes de 1000 [m], 2000 [m] et 3000 [m].
#
#    > **Indication** : il peut être utile de passer `xticks=0:1:20` en argument à la fonction `plot` pour simplifier cette estimation.

# +
Cᵈ, Δ = .75, .01
μs = LinRange(0, 30, 200)
# -

# 10. Pour un coefficient de trainée donné,
#     on veut calculer la masse de carburant minimale nécessaire en vue d'atteindre une altitude $H$ donnée,
#     en vue d'y déposer du matériel météorologique.
#     Pour ce faire,
#     on se propose de mettre en œuvre l'algorithme de Newton-Raphson sur une fonction scalaire prenant comme argument $\mu_0$
#     et s'annulant lorsque l'estimation de l'altitude obtenue par la fonction `max_altitude` est égale à $H$.
#     La méthode de Newton-Raphson nécessite de connaître la dérivée de la fonction dont on cherche une racine,
#     qui peut être obtenue par différentiation automatique.
#     On donne ci-dessous la structure `D` de nombre dual avec presque toutes les fonctions suffisantes pour entreprendre la résolution de l'équation différentielle,
#     sauf la fonction `tanh(x::D)` qui est à définir.

# +
import Base: +, -, *, /, ==, ≈, cos, sin, tanh, inv, conj, abs, isless, AbstractFloat, convert, promote_rule, show
struct D <: Number
    f::Tuple{Float64,Float64}
end
+(x::D, y::D)                           = D(x.f .+ y.f)
-(x::D, y::D)                           = D(x.f .- y.f)
-(x::D)                                 = D(.-(x.f))
*(x::D, y::D)                           = D((x.f[1]*y.f[1], (x.f[2]*y.f[1] + x.f[1]*y.f[2])))
/(x::D, y::D)                           = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
==(x::D, y::D)                          = x.f[1] == y.f[1] && x.f[2] == y.f[2]
≈(x::D, y::D)                           = x.f[1] ≈ y.f[1] && x.f[2] ≈ y.f[2]
abs(x::D)                               = D((abs(x.f[1]), x.f[2]*sign(x.f[1])))
abs2(x::D)                              = D((abs2(x.f[1]), 2x.f[1]*x.f[2]))
isless(x::D, y::D)                      = isless(x.f[1], y.f[1])
isless(x::D, y::Real)                   = isless(x.f[1], y)
isless(x::Real, y::D)                   = isless(x, y.f[1])
D(x::D)                                 = x
AbstractFloat(x::D)                     = x.f[1]
convert(::Type{D}, x::Real)             = D((x,zero(x)))
convert(::Type{Real}, x::D)             = x.f[1]
promote_rule(::Type{D}, ::Type{<:Real}) = D
show(io::IO,x::D)                       = print(io,x.f[1],x.f[2]<0 ? " - " : " + ",abs(x.f[2])," ε")
ε = D((0, 1))

tanh(x::D) = D((tanh(x.f[1]), 1 / cosh(x.f[1])^2 * x.f[2]))
@mark tanh(ε) ≈ ε
@mark tanh(1000 + ε) == 1
@mark tanh(-1000 + ε) == -1
@mark tanh(log(2) + ε) ≈ 3/5 + 16/25*ε
# -

# 11. Ecrire une fonction `newton_raphson_dual(f, x; maxiter = 100, δ = 1e-12)` de résolution par Newton-Raphson d'une équation scalaire `f(x) = 0`,
#     dans laquelle la dérivée de `f` au point courant est obtenue par exploitation des nombres duaux,
#     avec un nombre d'itérations maximal `maxiter` ($100$ par défaut) et une tolérance `δ` ($10^{-12}$ par défaut) pour un critère d'arrêt $|f(x)| < δ$.
#
#     > **Indication :** à chaque itération de la résolution, les valeurs de `f` et de sa dérivée en `x` peuvent être extraites du calcul de `y = f(x + D((0,1)))`.

# +
function newton_raphson_dual(f, x, maxiter=100; δ = 1e-12)

end

@mark newton_raphson_dual(x -> x^2 - 2, 1) ≈ √2
@mark newton_raphson_dual(x -> x^2 - 2, -1) ≈ -√2
@mark newton_raphson_dual(x -> x^3 - 2, 1) ≈ cbrt(2)
@mark newton_raphson_dual(x -> tanh(x) - .5, 1) ≈ atanh(.5)
# -

# 12. Écrire une fonction `find_fuel(H, Δ, Cᵈ, μ₀)` renvoyant la masse de carburant minimale requise pour atteindre une altitude $H$.
#     Calculer une estimation des quantités de carburant permettant d'atteindre des altidudes de 1000 [m], 2000 [m] et 3000 [m] pour `Cᵈ = .75`,
#     et tracer les courbes d'altitude correspondantes.

# +
function find_fuel(H, Δ, Cᵈ, μ₀)

end

Δ, Cᵈ = .01, .75
# Find values of μ₀ and plot here

@mark find_fuel(1000, Δ, Cᵈ, 5) |> floor == 3
@mark find_fuel(2000, Δ, Cᵈ, 5) |> floor == 7
@mark find_fuel(3000, Δ, Cᵈ, 5) |> floor == 12
# -
