# Discrete Mathematics Lecture Notes

## 1. Propositional Logic

### Proposition : 명제 / Propositional Logic : 명제 논리
- Truth Value : 진리값
- Truth Table : 진리표
- Atomic Propostition : 단일 명제
- Compound Proposition : 복합 명제
- Satisfiable : 만족 가능 = 참이 될 수 있음
- Unsatisfiable : 만족 불가능 = 참이 될 수 없음
### 논리 비교
- Logically Equivalent : 논리적 동치
- Tautology : 항진명제
- Contradiction : 모순
- Contigency : 불확정명제
### Operator : 연산자
- Negation : 부정
- Conjunction : 논리곱
- Disjunction : 논리합
- Exclusive Or : 배타적 논리합
- Implication : 함축(조건문)
    - Converse : 역
    - Contrapositive : 대우
    - Inverse : 이
    - Biconditional : 상호 조건문
- 우선순위 : ¬ > ∧ > ∨ > → > ⇔
### 주요 명제 논리 법칙
- De Morgan's Law : 드 모르간 법칙
- Identity Law : 항등 법칙
- Domination Law : 지배 법칙
- Idempotent Law : 등멱 법칙
- Double Negation Law : 이중 부정 법칙
- Commutative Law : 교환 법칙
- Associative Law : 결합 법칙
- Distributive Law : 분배 법칙
### Predicate : 술어 / Predicate Logic : 술어 논리
- Variable : 변수
- Predicate P(x) → Propositional Function / Variable a → Proposition / P(a) → Proposition
### Quantifier : 양화사, 한정 기호
- Universal Quantifier : 전칭 한정기호
    - Domain : 정의역 = Universe of Discourse : 논의 영역
    - Counter Example : 반례
- Existential Quantifier : 존재 한정기호
- Uniqueness Quantifier : 유일 한정기호
- 우선순위 : ∀, ∃ > 기타 논리 연산자
### Rules of Inference : 추론 규칙
- Premise : 전제, Conclusion : 결론
- Modus Ponens : 긍정 논법 = Elimination Implication : 조건문 기호 제거
- Modus Tollens : 부정 논법 = Denying the Consequent : 후건 부정
- Hypothetical Syllogism : 가설적 삼단논법
- Disjunctive Syllogism : 논리적 삼단논법 (∨ 제거)
- Addition : 가산 논법 (∨ 도입)
- Simplification : 단순화 논법 (∧ 제거)
- Conjunction : 논리곱 논법 (∧ 도입)
- Resolution : 융해법
### 양화 논리 법칙
- Universal Instantiation (UI) : 전칭 예시화 (∀ 제거)
- Universal Generalization (UG) : 전칭 일반화 (∀ 도입)
- Existential Instantiation (EI) : 존재 예시화 (∃ 제거)
- Existential Generalization (EG) : 존재 일반화 (∃ 도입)
### Proof : 증명
- Theorem : 정리
    - Definition : 정의
    - Axion (Postulate) : 공리
    - Lemma : 보조정리
    - Corollary : 따름정리
    - Conjecture : 가설
- Trivial Proof : 자명한 증명 (p → q에서 q가 항상 참)
- Vacuous Proof : 공허한 증명 (p → q에서 p가 항상 거짓)
- Direct Proof : 직접 증명
- Proof by Contraposition : 대우에 의한 증명
- Proof by Contradiction : 모순에 의한 증명 (귀류법)
- Proof of Equivalence : 동치 증명
- Existence Proof : 존재 증명
    - Constructive Proof : 생산적 증명 (존재의 증명과 동시에 해를 제시)

## 2. Basic Structures

### Set : 집합
- Element : 원소 = Member : 구성원
- 종류
    - Universal Set : 전체집합
    - Empty Set (Null Set) : 공집합
    - Singleton : 단일 원소 집합
    - Subset : 부분집합
    - Proper Subset : 진부분집합
    - Power Set : 멱집합
- 표기법
    - Roster Method : 원소나열법
    - Set-Builder Notation : 조건제시법
- Truth Set : 진리집합
- Cartesian Product : 데카르트곱, set of all ordered pairs
- 연산
    - Union : 합집합
    - Intersection : 교집합
    - Complement : 여집합
    - Difference : 차집합
- 연산 법칙
    - Identity Law : 항등 법칙 (항등원과 연산)
    - Domination Law : 지배 법칙 (항등원의 여집합과 연산)
    - Idempotent Law : 멱등 법칙 (같은 것끼리 연산)
    - Complementation Law : 보원 법칙 (여집합의 여집합)
    - Commutative Law : 교환 법칙
    - Associative Law : 결합 법칙
    - Distributive Law : 분배 법칙
    - De Morgan's Law : 드 모르간 법칙
    - Absorption Law : 흡수 법칙 (교집합과 합집합, 합집합과 교집합)
    - Complement Law : 보수 법칙 (여집합과 연산)
- Set Cardinality : 집합의 크기 / 기수
    - |S| : 무한집합 S의 기수
    - If there is a one-to-one function from A to B, |A| ≤ |B|
    - Cantor-Schröder–Bernstein Theorem : |A| ≤ |B| & |B| ≤ |A| → |A| = |B|
    - Countable Set : 셀 수 있는 무한집합 = N과 같은 기수(=$\aleph_0$)를 갖는 집합
    - Q is countable, R is uncountable
### Function : 함수
- Domain : 정의역
- Codomain : 공역
- Image : 상
- Preimage : 원상
- Range : 치역 = Image of Subset
- 종류
    - One-to-One Function (Injective Function) : 단사함수 (일대일함수)
    - Onto Function (Surjective Function) : 전사함수
    - Bijection (Bijective Function) : 전단사함수 (일대일대응) = One-to-One Correspondence
    - Identity Function : 항등함수
    - Inverse Function : 역함수
    - Composition : 합성함수
### Sequence : 수열
- 종류
    - Arithmetic Progression : 등차수열
        - Initial Term : 초항
        - Common Difference : 공차
    - Geometric Progression : 등비수열
        - Initial Term : 초항
        - Common Ratio : 공비
- Recurrence Relation : 점화 관계
- Closed Formula : 닫힌 공식 (일반항)
### Matrix : 행렬
- Identity Matrix : 항등행렬
- Zero-One Matrix : 0과 1만으로 이루어진 행렬
- Join : $A \vee B = \left[a_{ij} \vee b_{ij}\right]$
- Meet : $A \wedge B = \left[a_{ij} \wedge b_{ij}\right]$
- Boolean Product : $A \odot B = \left[\bigvee_{l=1}^{k}\left(a_{il}\wedge b_{lj}\right)\right]$
    - $A \odot I_n = I_m \odot A = A$
- $A^{r} = A \times A \times \cdots \times A$, $A^{[r]} = A \odot A \odot \cdots \odot A$

## 5. Induction and Recursion

### Mathematical Induction : 수학적 귀납법
- Basis Step & Inductive Step
- Inductive Hypothesis (IH)
- Strong indcution
### Well Ordering Property : 정렬성
### Recursively Defined Functions : 재귀함수
- Basis Step & Recursive Step
- Recursively Defined Sets
    - Structural Induction
- Recursive Algorithms
### Rooted Tree
- Full Binary Tree

## 6. Counting

### Basics of Counting
- The Product Rule
- The Sum Rule
- The Subtraction Rule
- The Division Rule
### Pigeonhole Principle : 비둘기집의 원리
- Generalized Pigeonhole Principle : When there are $N$ objects placed into $k$ boxes, there is at least one box containing at least $\left\lceil{N/K}\right\rceil$ objects.
### Ramsey Numbers
- Minimum number of people where there are $m$ mutual friends or $n$ mutual enemies.
### Permutation : 순열
### Combination : 조합
- Pascal's Identity : $C(n+1, r) = C(n, r-1) + C(n, r)$
- Vandermonde's Identity : $C(m+n, r) = \sum_{k=0}^{r}{C(m, r-k)\cdot C(n, k)}$
- $C(n+1, r+1) = \sum_{k=r}^{n}{C(k, r)}$
### Binomial Theorem : 이항정리

## 7. Discrete Probability

### Bayes' Theorem
- $p\left(F_j ~|~ E\right) = \dfrac{p\left(F_j\right)p\left(E ~|~ F_j\right)}{\sum_{i=1}^{n}p\left(F_i\right)p\left(E ~|~ F_i\right)}$
### Geometric Distribution
- $p(X=j) = (1-p)^{j-1} \cdot p$
- $E(X) = \dfrac{1}{p}$
### Chebyshev' Inequality
- $p\left(\left|X(s) - \dfrac{n}{2}\right| \geq \sqrt{n} \right) \leq \dfrac{V(X)}{r^2}$

## 8. Advanced Counting Techniques

### Recurrence Relation : 점화식
### Closed Formula : 일반항
### Linear Homogeneous Recurrence Relation
- $a_n = c_1\cdot a_{n-1} + c_2\cdot a_{n-2} + \cdots + c_k\cdot a_{n-k} \;\; (c_k \neq 0)$
- Characteristic Equation : $r^k - c_1\cdot r^{k-1} - \cdots - c_k = 0$
- When the characteristic equation has $t$ distinct roots $r_1, r_2, \cdots, r_t$ with multiplicities $m_1, m_2, \cdots, m_t \;\; (\sum{m_i} = k)$, the solution is the following.
$$
\begin{align*}a_n = &\left(\alpha_{1,0} + \alpha_{1,1}n + \cdots + \alpha_{1,m_{1}-1}n^{m_1-1}\right)r_1^n \\ &+ \left(\alpha_{2,0} + \alpha_{2,1}n + \cdots + \alpha_{2,m_2-1}n^{m_2-1}\right)r_2^n \\ &+  \cdots + \left(\alpha_{t,0} + \alpha_{t,1}n + \cdots + \alpha_{t,m_t-1}n^{m_t-1}\right)r_t^n\end{align*}
$$
    
### Linear Nonhomogeneous Recurrence Relation
- $a_n = c_1\cdot a_{n-1} + c_2\cdot a_{n-2} + \cdots + c_k\cdot a_{n-k} + F(n) \;\; (c_k \neq 0, \; F(n) \neq 0)$
- When $\{p_n\}$ is a particular solution and $\{h_n\}$ is a solution of the associated homogenous recurrence relation, every solution is in the form of $\{p_n + h_n\}$
### Dynamic Programming : 동적 계획법
- Breaks down into simpler overlapping subproblems.
### Divide and Conquer : 분할정복
- Recursively divides a problem into a fixed number of nonoverlapping subproblems.
- Divide and Conquer Recurrence Relation : $T(n) = aT(n/b) + f(n)$
- Size(Complexity) of Algorithm
    - "Even More: Simplified Master Theorem : $T(n) = aT(n/b) + c \;\; \left(a \geq 1, \; b \in \mathbb{N}^{+}, \; b ~|~ n, \; c > 0\right)$
        - $T(n) = \begin{cases}O\left(n^{\log_ba}\right) & (a > 1) \\ O(\log n) & (a = 1)\end{cases}$
        - When $n = b^k, \; a \neq 1,\; k \in \mathbb{N}^+$,  $T(n) = C_1\cdot n^{\log_ba} + C_2 \;\; \left(C_1 = f(1) + \dfrac{c}{a-1}, \; C_2 = -\dfrac{c}{a-1}\right)$
    - Simplified Master Theorem : $T(n) = aT(n/b) + cn^d \;\; \left(a \geq 1, \; b \in \mathbb{N}^+, \; b ~|~ n, \; c > 0, \; d \geq 0\right)$
        - $T(n) = \begin{cases}O\left(n^d\right) & \left(a < b^d\right) \\ O(n^d\log{n}) & \left(a = b^d\right) \\ O\left(n^{\log_ba}\right) & \left(a > b^d\right)\end{cases}$
### Principle of Inclusion-Exclusion
- When $A_1, A_2, \cdots , A_n$ are finite sets, then :
$$
\left|A_1 \cup A_2 \cup \cdots \cup A_n\right| = \sum_{i=1}^{n}\left|A_i\right| - \sum_{1\leq i < j \leq n}\left|A_i \cap A_j\right| + \sum_{1 \leq i < j < k \leq n}\left|A_i\cap A_j\cap A_k\right| - \cdots + (-1)^{n-1}\left|A_1 \cap \cdots \cap A_n\right|
$$
$$
\begin{align*}\left|\bigcup_{i=1}^{n}A_i\right| &= \sum_{k=1}^{n}(-1)^{k-1}\left(\sum_{1\leq i_1<\cdots < i_k \leq n}\left|A_{i_1} \cap \cdots \cap A_{i_k}\right|\right) \\ &= \sum_{\emptyset \neq J \subseteq \{1, \cdots , n\}}(-1)^{\left|J\right|-1} \left|\bigcap_{j \in J}A_j\right|\end{align*}
$$

## 9. Relations

### Binary Relations
- Reflexivity
- Symmetricity
- Antisymmetricity
- Transitivity
### Composition : $S \circ R$
- Power of Relation : $R^n$
- Relation $R$ on $A$ is transitive. ⇔ $(\forall n) \; R^n \subseteq R$
### Closures
- Reflexive Closure : $S = R \cup \Delta ~~ (\Delta = \{(a, a) ~|~ a \in A\})$
- Symmetric Closure : $S = R \cup R^{-1} ~~ (R^{-1} = \{(b, a) ~|~ (a, b) \in R\})$
- Antisymmetric Closure
- Transitive Closure : $S = R^* = \bigcup_{k=1}^{n}R^k$
### Paths
- There is a path of length $n \; (n \in \mathbb{N}^+)$ from $a$ to $b$. ⇔ $(a, b) \in R^{n}$
- If there is a path from $a$ to $b$, there is such a path with length not exceeding $n$. If $a \neq b$, there is such a path with length not exceeding $n-1$.
### Connectivity Relation : $R^*$
- Consists of pairs $(a, b)$ where a path from $a$ to $b$ exists in $R$.
- $R^* = \bigcup_{n=1}^{\infty}R^n$
### Equivalence Relation : $a \sim b$
- Relation is reflexive, symmetric, and transitive.
- Elements related on a equivalence relation are called 'equivalent'
- Equivalence Classes : $[a]_R ~~ (= [a])$
    - The set of all elements that are equivalent to an element $a$.
    - $(a, b) \in R \Leftrightarrow [a] = [b] \Leftrightarrow [a] \cap [b] \neq \emptyset$
### Partition
- A collection of disjoint nonempty subsets of $A$ that have $A$ as their union.
    - $A_i \neq \emptyset ~~ (i \in I)$
    - $A_i \cap A_j \neq \emptyset ~~ (i \neq j)$
    - $\bigcup_{i \in I} A_i = A$
- Equivalence Classes form a partition of $A$ : $\bigcup_{a \in A} [a] = A$
### Partial Orderings
- Relation is reflexive, antisymmetric, and transitive.
- A set with a partial ordering is called a partially ordered set (poset), denoted by $(S, R)$.
- $a \preccurlyeq b \Leftrightarrow (a, b) \in R$, when $R$ is a partial ordering.
### Comparability
- $a, b \in S$ are comparable if either $a \preccurlyeq b$ or $b \preccurlyeq a$ ⇔ $(a, b) \in R$ or $(b,a) \in R$
- Total Order
    - If $(S, \preccurlyeq)$ is a poset and every two elements of $S$ are comparable, $S$ is called a totally ordered set and $\preccurlyeq$ is called a total order.
### Well-Ordering Principle
- When $S$ is a totally ordered set and every nonempty subset of $S$ has a least element.
- Well-Ordered Induction
### Hasse Diagrams

## 10. Graphs

### Terminology
- Vertices (Nodes)
- Edges
- Endpoints
- SImple Graph
- Multigraph
- Pseudograph
- Simple Directed Graph
- Directed Multigraph
- Adjacent (Neighbor)
- Incident
- Initial Vertex, End Vertex (Terminal)
- Neighborhood
- Degree
- In-Degree, Out-Degree
### The Handshaking Theorem
- $2|E| = \sum_{v \in V}\mathrm{deg}(v)$
- $\mathrm{deg}^{-}(v) = \mathrm{deg}^{+}(v) = |E|$
### Types of Graphs
- Complete Graph
- Cycle
- Bipartite Graph
- Complete Bipartite Graph
- Matching
- Complete Matching
### Hall's Theorem
- A bipartite graph $G$ with bipartition $(V_1, V_2)$ has a complete matching from $V_1$ to $V_2$ iff $|N(S)| \geq |S|$ for all subsets $S \subseteq V_1$.
### New Graphs
- Subgraph
- Proper Subgraph
### Adjacencey Matrices
### Isomorphism
- Graph Invariant
### Paths
- Directed or Undirected
- Simple Path
- Circuit
- Connected Graph : Path exists between every pair of vertices (⇔ Disconnected)
    - Connected Components : A connected subgraph  that is not a proper subgraph of another connected subgraph (maximal connected subgraphs)
- Strongly Connected : Path exists between every pair of vertices, both ways
- Weakly Connected : Path exists betwen every pair of vertices, either way
    - Strongly Connected Components : The maximal strongly connected subgraphs
### Multigraph Model
- Euler Circuit
    - Each vertices should have even degree
- Euler Path
    - Has exactly two vertices of odd degree : Has Eular Path but not an Euler Circuit
- Hamilton Circuit
    - Dirac Theorem : If degree of every vertex is at least $n/2$, graph has Hamilton Circuit.
    - Ore Theorem : If sum of degree of any nonadjacent vertices is at least $n$, graph has Hamilton Circuit.
- Hamilton Path