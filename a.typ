Функция потерь Хьюбера:

$
R(x) = frac(x^2, 2) I(|x| <= c) + c (|x| - frac(c, 2)) I(|x| > c).
$

Её производная имеет вид:

$
R'(x) = cases(
  x, & |x| <= c, \\
  c op("sign")(x), & |x| > c.
)
$

Для задачи

$
L(theta) = sum_(i=1)^n R(Y_i - x_i^T theta) -> min_(theta in RR^d)
$

обозначим остаток на объекте $i$ как

$
r_i(theta) = Y_i - x_i^T theta.
$

Тогда градиент целевой функции:

$
nabla_theta L(theta) = -sum_(i=1)^n x_i R'(r_i(theta)).
$

При шаге $eta > 0$ итерация градиентного спуска имеет вид:

$
theta^(k+1) = theta^k + eta sum_(i=1)^n x_i R'(Y_i - x_i^T theta^k).
$

Подставляя производную функции Хьюбера:

$
theta^(k+1) = theta^k + eta sum_(i=1)^n x_i cases(
  Y_i - x_i^T theta^k, & |Y_i - x_i^T theta^k| <= c, \\
  c op("sign")(Y_i - x_i^T theta^k), & |Y_i - x_i^T theta^k| > c.
).
$

= Матричная запись

Пусть $X in RR^(n times d)$ — матрица с $i$-й строкой $x_i^T$,
а $Y = (Y_1, dots, Y_n)^T$ — вектор ответов.
Тогда вектор остатков:

$
r(theta) = Y - X theta.
$

Производная Хьюбера выражается через обрезание до интервала $[-c, c]$:

$
R'(r) = op("clip")(r, -c, c) = min(c, max(-c, r)).
$

Для вектора операция $op("clip")$ применяется покомпонентно.
Градиент и шаг градиентного спуска принимают вид:

$
nabla_theta L(theta) = -X^T op("clip")(Y - X theta, -c, c),
$

$
theta^(k+1) = theta^k + eta X^T op("clip")(Y - X theta^k, -c, c).
$

Если все остатки по модулю не превосходят $c$, шаг совпадает с шагом для квадратичной функции потерь:

$
theta^(k+1) = theta^k + eta X^T (Y - X theta^k).
$

Множителя $1/n$ нет, поскольку целевая функция — сумма, а не среднее потерь.
