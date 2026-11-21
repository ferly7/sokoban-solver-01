# Sokoban solver — sokoban-solver-01

A push-optimal Sokoban solver in a single TypeScript file (`main.ts`), run
with Node 22: `node main.ts` reads every instance under
`$MAKEFUN_INPUT_DIR` (default `/input`), each an XSB `level.txt`, and writes
the player's moves to `$MAKEFUN_OUTPUT_DIR/<id>/solution.txt` (default
`/output`).

## Approach

Levels from this generator are a one-square-wide hallway with small rooms
(≤ 6×6, 3–4 boxes each) opening off it. A box pushed out of its room can
never return, so the rooms are independent sub-problems:

1. Split the level into rooms by cutting the hallway's dead squares (squares
   from which a lone box can reach no goal, found by pulling back from the
   goals) out of the player's row.
2. Solve each room with a breadth-first search over pushes — uniform cost,
   so the first solution is push-optimal — with the other rooms' boxes
   standing as walls. States are canonicalised as the sorted box squares
   plus the lowest square of the player's reachable region; dead squares and
   frozen 2×2 blocks are pruned.
3. Concatenate the rooms' move strings; the player walks on from wherever
   the previous room left them.

A level that is not a set of rooms falls back to the same push search over
the whole level, then to a guided A* over pushes. All budgets are state
counts, never the clock, so runs are deterministic.
