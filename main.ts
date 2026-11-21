// Sokoban solver: push-optimal search, room by room.
//
// A level from this generator is a one-square-wide hallway with small rooms
// (at most 6x6, three or four boxes each) opening off it. A box pushed out of
// its room can never come back, so the rooms are independent: solve each one
// with a breadth-first search over pushes (uniform cost, so the first
// solution found is push-optimal) while the other rooms' boxes stand as
// walls, then walk the player on to the next room. The pushes add up to the
// level's optimum.
//
// The push search canonicalises a state as the sorted box squares plus the
// lowest square of the player's reachable region, and prunes two deadlocks:
// a box on a dead square (one no push can bring to any goal, found by pulling
// back from the goals) and a frozen 2x2 block of walls and boxes holding a
// box off its goal. If a level is not a set of rooms, the same search runs on
// the whole level, and a guided A* over pushes is the last resort. Every
// budget below is a count of states, never the clock, so two runs of this
// program write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK_CH = ['u', 'd', 'l', 'r'];
const PUSH_CH = ['U', 'D', 'L', 'R'];

const ROOM_STATE_CAP = 2_000_000;
const LEVEL_STATE_CAP = 1_500_000;
const ASTAR_STATE_CAP = 1_500_000;

interface Level {
  w: number;
  h: number;
  size: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever the text shows: nothing leaves the board.
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, size, wall, goals, boxes, player };
}

/**
 * For every square, the fewest pushes that bring a lone box from it to some
 * goal, ignoring the other boxes (the player walks freely). Squares that
 * cannot reach a goal keep Infinity: a box there is dead. Computed by pulling
 * back from all goals at once: a box at y came from x = y - d when both x and
 * the square the player stood on, x - d, are open.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Float64Array {
  const dist = new Float64Array(wall.length).fill(Infinity);
  const queue: number[] = [];
  for (const g of goals) {
    dist[g] = 0;
    queue.push(g);
  }
  for (let head = 0; head < queue.length; head++) {
    const y = queue[head];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * w + DX[k];
      const x = y - d;
      if (x < 0 || x >= wall.length) continue;
      if (wall[x] || wall[x - d] || dist[x] !== Infinity) continue;
      dist[x] = dist[y] + 1;
      queue.push(x);
    }
  }
  return dist;
}

/** A board view for one search: walls (with foreign boxes folded in), one set of goals, its boxes. */
interface SearchBoard {
  w: number;
  size: number;
  wall: Uint8Array;
  isGoal: Uint8Array;
  dist: Float64Array; // lone-box push distance to the nearest of this board's goals
}

/**
 * Scratch for flood fills, shared by every search on one board size. `seen`
 * is stamped rather than cleared, so a fill costs only the squares it visits.
 */
class Scratch {
  seen: Int32Array;
  stack: Int32Array;
  stamp = 0;
  constructor(size: number) {
    this.seen = new Int32Array(size);
    this.stack = new Int32Array(size);
  }
  /** Flood from `from` around walls and boxes; returns the region's lowest square. */
  regionLowest(board: SearchBoard, occupied: Uint8Array, from: number): number {
    const { w, wall } = board;
    const st = ++this.stamp;
    let n = 0;
    let low = from;
    this.stack[n++] = from;
    this.seen[from] = st;
    while (n > 0) {
      const p = this.stack[--n];
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k] * w + DX[k];
        if (wall[q] === 0 && occupied[q] === 0 && this.seen[q] !== st) {
          this.seen[q] = st;
          this.stack[n++] = q;
        }
      }
    }
    return low;
  }
  /** Whether `target` lies in the region the last regionLowest call filled. */
  inRegion(target: number): boolean {
    return this.seen[target] === this.stamp;
  }
}

/** Whether the box just pushed to `t` froze: a 2x2 of walls and boxes around it, some box off its goal. */
function frozenAround(board: SearchBoard, occupied: Uint8Array, t: number): boolean {
  const { w, wall, isGoal } = board;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    if (a < 0) continue;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !isGoal[a]) ||
      (occupied[b] && !isGoal[b]) ||
      (occupied[c] && !isGoal[c]) ||
      (occupied[d] && !isGoal[d])
    ) {
      return true;
    }
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, as lower-case LURD letters. */
function walkPath(
  board: SearchBoard,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const { w, wall } = board;
  const prev = new Int32Array(wall.length).fill(-1);
  const letter: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      letter[q] = WALK_CH[k];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkPath: target out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = letter[p] + s;
  return s;
}

/** The square the player stands on after playing `moves` from `from`. */
function endSquare(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

interface PushNode {
  boxes: number[];
  player: number; // the player's square in this state (where the last push left them)
  parent: number;
}

/** Rebuild the move string of state `found` from the node list. */
function rebuildMoves(board: SearchBoard, nodes: PushNode[], found: number, startPlayer: number): string {
  const chain: number[] = [];
  for (let s = found; s > 0; s = nodes[s].parent) chain.push(s);
  chain.reverse();
  const occupied = new Uint8Array(board.size);
  let moves = '';
  let player = startPlayer;
  let boxes = nodes[0].boxes;
  for (const s of chain) {
    const next = nodes[s].boxes;
    let from = -1;
    let to = -1;
    for (const b of boxes) if (!next.includes(b)) from = b;
    for (const b of next) if (!boxes.includes(b)) to = b;
    const d = to - from;
    const k = DY.findIndex((dy, i) => dy * board.w + DX[i] === d);
    for (const b of boxes) occupied[b] = 1;
    moves += walkPath(board, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH_CH[k];
    player = from;
    boxes = next;
  }
  return moves;
}

/**
 * Breadth-first search over pushes: push-optimal. A state is the sorted box
 * squares plus the lowest square of the player's reachable region. Returns
 * the move string, or null when the state cap runs out or no solution exists.
 */
function solvePushBfs(
  board: SearchBoard,
  startBoxes: number[],
  startPlayer: number,
  cap: number,
): string | null {
  const { size, wall, isGoal, dist } = board;
  if (startBoxes.every((b) => isGoal[b])) return '';
  if (startBoxes.some((b) => dist[b] === Infinity)) return null;
  const scratch = new Scratch(size);
  const occupied = new Uint8Array(size);
  const numericKey = startBoxes.length <= 5;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (numericKey) {
      let key = low;
      for (const b of boxes) key = key * size + b;
      return key;
    }
    return `${low}|${boxes.join(',')}`;
  };
  const nodes: PushNode[] = [{ boxes: startBoxes.slice(), player: startPlayer, parent: -1 }];
  const index = new Map<number | string, number>();
  for (const b of startBoxes) occupied[b] = 1;
  index.set(keyOf(startBoxes, scratch.regionLowest(board, occupied, startPlayer)), 0);
  for (const b of startBoxes) occupied[b] = 0;
  for (let head = 0; head < nodes.length; head++) {
    const { boxes, player } = nodes[head];
    for (const b of boxes) occupied[b] = 1;
    scratch.regionLowest(board, occupied, player);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * board.w + DX[k];
        const target = b + d;
        if (!scratch.inRegion(b - d)) continue;
        if (wall[target] || occupied[target] || dist[target] === Infinity) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenAround(board, occupied, target);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((x, y) => x - y);
          key = keyOf(next, scratch.regionLowest(board, occupied, b));
          // regionLowest overwrote the stamp; refill the current region for the next direction.
          scratch.regionLowest(board, occupiedWith(occupied, target, b), player);
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen || next === null || index.has(key)) continue;
        if (nodes.length >= cap) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, nodes.length);
        nodes.push({ boxes: next, player: b, parent: head });
        if (next.every((x) => isGoal[x])) {
          for (const x of boxes) occupied[x] = 0;
          return rebuildMoves(board, nodes, nodes.length - 1, startPlayer);
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/** `occupied` with the box moved back from `target` to `from` (for refilling a region mid-expansion). */
function occupiedWith(occupied: Uint8Array, target: number, from: number): Uint8Array {
  occupied[target] = 0;
  occupied[from] = 1;
  return occupied;
}

/**
 * Last-resort guided search over pushes: A* on pushes made plus the summed
 * lone-box push distances. Not guaranteed optimal; it exists so an
 * off-structure level still gets a valid answer instead of none.
 */
function solvePushAstar(
  board: SearchBoard,
  startBoxes: number[],
  startPlayer: number,
  cap: number,
): string | null {
  const { size, wall, isGoal, dist } = board;
  if (startBoxes.every((b) => isGoal[b])) return '';
  if (startBoxes.some((b) => dist[b] === Infinity)) return null;
  const estimate = (boxes: number[]): number => boxes.reduce((a, b) => a + dist[b], 0);
  const scratch = new Scratch(size);
  const occupied = new Uint8Array(size);
  const nodes: (PushNode & { g: number })[] = [
    { boxes: startBoxes.slice(), player: startPlayer, parent: -1, g: 0 },
  ];
  const best = new Map<string, number>();
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;
  for (const b of startBoxes) occupied[b] = 1;
  best.set(keyOf(startBoxes, scratch.regionLowest(board, occupied, startPlayer)), 0);
  for (const b of startBoxes) occupied[b] = 0;
  // Binary heap of node indexes on f = g + estimate, ties to the deeper node.
  const heap: number[] = [0];
  const fOf = (i: number): number => nodes[i].g + estimate(nodes[i].boxes);
  const before = (a: number, b: number): boolean => {
    const fa = fOf(a);
    const fb = fOf(b);
    return fa < fb || (fa === fb && nodes[a].g > nodes[b].g);
  };
  const heapPush = (i: number): void => {
    heap.push(i);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (!before(heap[c], heap[p])) break;
      [heap[c], heap[p]] = [heap[p], heap[c]];
      c = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  while (heap.length > 0) {
    const cur = heapPop();
    const { boxes, player, g } = nodes[cur];
    if (boxes.every((b) => isGoal[b])) return rebuildMoves(board, nodes, cur, startPlayer);
    if (nodes.length >= cap) return null;
    for (const b of boxes) occupied[b] = 1;
    scratch.regionLowest(board, occupied, player);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * board.w + DX[k];
        const target = b + d;
        if (!scratch.inRegion(b - d)) continue;
        if (wall[target] || occupied[target] || dist[target] === Infinity) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenAround(board, occupied, target);
        let key = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((x, y) => x - y);
          key = keyOf(next, scratch.regionLowest(board, occupied, b));
          scratch.regionLowest(board, occupiedWith(occupied, target, b), player);
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const known = best.get(key);
        if (known !== undefined && known <= g + 1) continue;
        best.set(key, g + 1);
        nodes.push({ boxes: next, player: b, parent: cur, g: g + 1 });
        heapPush(nodes.length - 1);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway —
 * a box there reaches no goal) out of the floor; each part left holding boxes
 * is a room. Null when the level is no set of rooms: a box in the hallway, a
 * goal cut off, or a part whose boxes and goals do not balance.
 */
function splitRooms(level: Level): Room[] | null {
  const { w, size, wall, goals, boxes, player } = level;
  const dist = pushDistances(w, wall, goals);
  const rowStart = Math.floor(player / w) * w;
  const cut = new Uint8Array(size);
  for (let i = rowStart; i < rowStart + w; i++) {
    if (!wall[i] && dist[i] === Infinity) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const parts: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const id = parts.length;
    const squares = [s];
    part[s] = id;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * w + DX[k];
        if (q < 0 || q >= size) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = id;
          squares.push(q);
        }
      }
    }
    parts.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) parts[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    parts[part[g]].goals.push(g);
  }
  if (parts.some((p) => p.boxes.length !== p.goals.length)) return null;
  return parts
    .filter((p) => p.boxes.length > 0)
    .map((p) => ({ boxes: p.boxes.slice().sort((a, b) => a - b), goals: p.goals }));
}

/** One search board for a goal set under a wall array. */
function searchBoard(level: Level, wall: Uint8Array, goals: number[]): SearchBoard {
  const isGoal = new Uint8Array(level.size);
  for (const g of goals) isGoal[g] = 1;
  return { w: level.w, size: level.size, wall, isGoal, dist: pushDistances(level.w, wall, goals) };
}

/** Solve one level text; returns the move string ('' when nothing better was found). */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const allGoals = searchBoard(level, level.wall, level.goals);
  if (level.boxes.every((b) => allGoals.isGoal[b])) return '';
  const rooms = splitRooms(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let solvedAll = true;
    const done: boolean[] = rooms.map(() => false);
    for (let ri = 0; ri < rooms.length; ri++) {
      const wall = level.wall.slice();
      rooms.forEach((other, oi) => {
        if (oi === ri) return;
        for (const b of done[oi] ? other.goals : other.boxes) wall[b] = 1;
      });
      const board = searchBoard(level, wall, rooms[ri].goals);
      const part = solvePushBfs(board, rooms[ri].boxes, player, ROOM_STATE_CAP);
      if (part === null) {
        solvedAll = false;
        break;
      }
      moves += part;
      player = endSquare(level.w, player, part);
      done[ri] = true;
    }
    if (solvedAll) return moves;
  }
  const whole = solvePushBfs(allGoals, level.boxes, level.player, LEVEL_STATE_CAP);
  if (whole !== null) return whole;
  return solvePushAstar(allGoals, level.boxes, level.player, ASTAR_STATE_CAP) ?? '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
