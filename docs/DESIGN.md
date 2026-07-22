# Ro — Design

Ro is a statically-typed Lisp for numerical computing. Three commitments define it:

1. **Whole-array semantics.** Rank-*n* arrays are the primitive value; operations broadcast or reduce over them. Users rarely write scalar loops.
2. **Inference-first static types.** Hindley–Milner with an effect row. Annotations are optional documentation, not obligations.
3. **Differentiation as a language feature.** `grad` is a Core-to-Core transform, so it composes with control flow, survives compilation, and nests.

## Pipeline

```
source
  │  Reader            megaparsec → Syntax (surface s-expressions)
  ▼
Syntax
  │  Expander          macros, quasiquote, derived forms → Syntax
  ▼
Syntax
  │  Infer             Algorithm W + effect rows → annotated Syntax
  ▼
Core                   typed, A-normal form, explicit effects
  │  Transform         AD, fusion, specialization: Core → Core
  ▼
Core
  ├─ Eval              phase 1: direct interpreter
  ├─ Compile → VM      phase 2: register bytecode
  └─ Compile → LLVM    phase 3: JIT for fused kernels
```

Core is the hinge. It is the only thing the backends, the optimizer, and AD agree on, so it must be small: literals, variables, application, lambda, `let`, `if`, primitive ops. Everything in the surface language reduces to it.

**Core is in ANF.** Every intermediate is named, every call operand is atomic. This is not stylistic — it is what makes the AD transform local (each `let` binding gets one tangent/adjoint rule) and what makes the bytecode register allocator trivial.

## Syntax

S-expressions, following the `hello_world.ro` sketch.

```lisp
(: name type)                     ; standalone type annotation
(define (f x y) body ...)         ; function definition
(define x expr)                   ; value definition
(fn (x y) body)                   ; lambda
(let ((x e) ...) body)            ; parallel binding
(let* ((x e) ...) body)           ; sequential binding
(if c t e)
(do e1 e2 ...)                    ; effectful sequencing
```

Literals: `42` (Int), `3.14` (Double), `"..."` (String, `$name` interpolation), `#t`/`#f`, `[1 2 3]` array, `[[1 2] [3 4]]` nested → rank 2.

Keyword arguments for array ops, since axis/shape parameters are positional-hostile:

```lisp
(sum a :axis 0)
(reshape a :shape [3 4])
(zeros :shape [3 3] :dtype 'f64)
```

**Macros.** Defer to phase 2. When they land, use explicit renaming (`syntax-rules`-style pattern matching, hygiene by renaming introduced binders). Do not ship unhygienic `defmacro`; it becomes impossible to remove later.

## Types

```
τ ::= Int | Double | Bool | String | Unit
    | (Arr n τ)            n = rank, a type-level natural
    | (-> τ ... ε τ)       ε = effect row
    | α                    type variable

ε ::= {}                   pure
    | {Io | ρ}             row with tail variable
```

`(: main (Io Unit))` from the example reads as: no arguments, effect row containing `Io`, result `Unit`.

**Rank is in the type; shape is not.** `(Arr 2 Double)` is a matrix — the typechecker rejects passing a vector where a matrix is expected, and knows statically how many indices an access takes. Dimension *sizes* are checked at runtime. Full shape typing (`(Arr [m k])` × `(Arr [k n])` → `(Arr [m n])`) needs a constraint solver over type-level naturals and is a multi-week project; rank gets most of the ergonomic benefit for a fraction of the cost. Design `Arr`'s representation so shape indices can be added later without a breaking change.

**Numerics: no tower.** `Int` is 64-bit; `Double` is IEEE 754 binary64. Literals default to `Int`, unify with `Double` when required by context, and never silently widen at runtime. A Scheme-style tower with promotion and bignums is the single most effective way to make numeric code slow, and it buys nothing for this domain.

**Effects.** Start with two: `Io` and `Alloc` (in-place array mutation). The row is polymorphic, so higher-order functions propagate effects instead of forcing them:

```lisp
(: map (-> (-> a e b) (Arr n a) e (Arr n b)))
```

`map` with a pure function stays pure. This is the whole reason to use rows rather than a fixed effect lattice.

## Evaluation semantics

**Strict, left-to-right, call-by-value.** Non-negotiable for numerics: laziness makes memory behavior and floating-point sequencing unpredictable, and thunk allocation dominates in tight code.

**Arrays are immutable by default.** Mutation is available inside an `Alloc` region, which lets the optimizer rewrite `(+ a b)` into an in-place update when `a` is provably dead. Users get value semantics; the compiler gets buffer reuse.

**Broadcasting.** NumPy rules — right-align shapes, dimensions of size 1 stretch, mismatches are runtime errors. Well understood, and the semantics people already expect.

## Automatic differentiation

`grad` is a Core→Core transform, run after typechecking and before backend selection.

- **Reverse mode** for `(grad f)` where `f : (Arr n Double) -> Double` — the loss-function case, the common one.
- **Forward mode** for `(jvp f)` and for the inner derivative in nested applications.
- ANF makes this mechanical: each `let x = prim(y, z)` emits a tape entry forward and one adjoint accumulation backward.
- Closure under transformation is the requirement that matters: the output of `grad` is ordinary Core, so `(grad (grad f))` type-checks and compiles like anything else.

Every primitive carries a derivative rule in the primitive table alongside its type and its evaluator. Adding a primitive without a rule should be a compile error in the Haskell source, not a runtime failure in Ro — encode this with a total record type, not a partial lookup.

`ad` should be removed from `build-depends`. It differentiates Haskell functions; Ro functions are data.

## Performance

The strategy is that **interpretation overhead amortizes over array size**, and the compiler's job is eliminating *intermediates*, not eliminating dispatch.

**Representation.** `Data.Vector.Unboxed` payload plus a shape record; strided views so `transpose`, `reshape`, and slicing are O(1) and allocation-free.

**Fusion is the highest-leverage optimization.** `(+ (* a b) c)` naively allocates a full temporary for `(* a b)`. A Core-level pass fuses elementwise chains into a single loop nest with one output buffer. On large arrays this is a 2–3× win from halved memory traffic, and it is available in phase 1 — it needs no codegen at all, just a rewrite over Core plus a fused-loop primitive.

**BLAS for the ops that deserve it.** `@` (matmul), factorizations, and solves go straight to OpenBLAS/LAPACK. Do not hand-write these; a naive triple loop is 50× off a tuned `dgemm` and no amount of your own optimizer closes that.

**Static types mean untagged registers.** Because inference has already resolved every type, phase 2's VM can use unboxed registers with no runtime tag checks and no boxing on the arithmetic path. This is the concrete performance payoff of choosing HM over dynamic typing, and it's worth designing the bytecode around from the start.

**Phase 3 (LLVM) is for scalar and loopy code only.** Whole-array code is already at memory bandwidth after fusion + BLAS; JIT adds nothing. Where it pays is user-written scalar loops and fused kernels with irregular control flow. Scope it that narrowly.

**Benchmarks from day one.** A `bench/` suite with a fixed set of kernels — elementwise chain, reduction, matmul, gradient of a small MLP, an N-body step — tracked per commit. Without it, "performance work" becomes untestable folklore.

## Roadmap

Each phase ends with something runnable.

### Phase 1 — Core language (interpreted)
Reader → Syntax → Infer → Core → Eval. Rank-typed arrays on unboxed vectors, elementwise fusion, BLAS matmul, reverse-mode `grad`, a REPL, `examples/hello_world.ro` running.

*Done when:* a linear-regression example trains via `grad` and gradient descent, and the fused elementwise benchmark shows the expected single-pass memory traffic.

### Phase 2 — Bytecode VM
Register bytecode with untagged registers, Core→bytecode compiler, hygienic macros, richer stdlib (slicing, `LAPACK` solves, random), user-facing error messages with source spans.

*Done when:* the scalar benchmarks are 10×+ over phase 1 and array benchmarks have not regressed.

### Phase 3 — Native codegen
LLVM JIT for fused kernels and scalar loops, forward mode + nested `grad`, in-place optimization via the `Alloc` effect, possibly shape types.

*Done when:* N-body and other loop-heavy benchmarks land within ~2× of equivalent C.

## Open questions

- **Modules.** Nothing above addresses namespacing. Needed before the stdlib grows past one file; worth deciding early since it touches the reader and the type environment.
- **Integer overflow.** Wrap, trap, or promote? Trapping is right for correctness, costs a branch per op. Probably: trap by default, `unsafe-` variants for kernels.
- **Complex numbers.** Cheap if designed into the dtype story now, invasive if retrofitted.
- **Shape polymorphism syntax.** Even deferred, reserve the syntax so `(Arr 2 Double)` can grow into `(Arr [m n] Double)` compatibly.
