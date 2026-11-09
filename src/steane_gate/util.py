import stim

# Stim gate names accepted by _Circuit.op, plus friendly aliases.
_ALIASES = {"ST": "S_DAG", "SDG": "S_DAG"}
_GATES = {"I", "X", "Y", "Z", "H", "S", "S_DAG", "R", "RX", "M", "MX", "MR", "MRX"}
_MEASUREMENTS = {"M", "MX", "MR", "MRX"}


class _Circuit:
    """Thin builder for a Stim circuit that keeps track of the measurement record."""

    def __init__(self, qubits: int):
        self.num_qubits = qubits
        self.lines = []
        self.num_measurements = 0
        # Call the base methods explicitly: subclasses override op() with logical semantics.
        _Circuit.op(self, "R", range(qubits))
        _Circuit.tick(self)

    def op(self, name, qubits):
        """
        Applies the single-qubit gate, reset or measurement `name` to each of
        `qubits`. 'ST' is accepted as an alias for Stim's 'S_DAG'.
        For measurements, returns the absolute record indices of the results,
        in the same order as `qubits`.
        """
        name = _ALIASES.get(name, name)
        if name not in _GATES:
            raise ValueError(f"Unsupported gate {name!r}")
        qubits = [int(q) for q in qubits]
        if not qubits:
            return []
        self.lines.append(f"{name} {' '.join(map(str, qubits))}")
        if name in _MEASUREMENTS:
            start = self.num_measurements
            self.num_measurements += len(qubits)
            return list(range(start, self.num_measurements))
        return None

    def cx(self, pairs):
        """
        Applies CNOT to each (control, target) pair. A qubit may appear in at
        most one pair, so all gates in one call act in parallel.
        """
        pairs = [(int(c), int(t)) for c, t in pairs]
        if not pairs:
            return
        used = [q for p in pairs for q in p]
        if len(used) != len(set(used)):
            raise ValueError("A qubit appears in more than one CX pair; split the call into layers")
        self.lines.append("CX " + " ".join(f"{c} {t}" for c, t in pairs))

    def tick(self):
        self.lines.append("TICK")

    def _rec(self, indices):
        return " ".join(f"rec[{i - self.num_measurements}]" for i in sorted(indices))

    def detector(self, indices, coords=()):
        """Declares that the XOR of the given measurement results is deterministic."""
        c = f"({', '.join(map(str, coords))})" if coords else ""
        self.lines.append(f"DETECTOR{c} {self._rec(indices)}")

    def observable(self, indices, index=0):
        """Adds the XOR of the given measurement results to logical observable `index`."""
        self.lines.append(f"OBSERVABLE_INCLUDE({index}) {self._rec(indices)}")

    def get_circuit(self) -> stim.Circuit:
        return stim.Circuit("\n".join(self.lines))


def get_schedule(pairs):
    """
    Splits CNOT `pairs` into the minimum number of parallel layers.

    Uses König's bipartite edge colouring, so it needs exactly Δ layers, where
    Δ is the largest number of pairs touching one qubit. The CNOT graph must be
    bipartite, which always holds for ancilla <-> data syndrome extraction.
    Pairs are only regrouped, so use it only for CNOTs that commute
    (no qubit is a control in one pair and a target in another).
    """
    pairs = [(int(u), int(v)) for u, v in pairs]
    side = {}
    for u, v in pairs:
        side.setdefault(u, 0)
        side.setdefault(v, 1 - side[u])
        if side[u] == side[v]:
            raise ValueError("CNOT graph is not bipartite")
    deg = {}
    for u, v in pairs:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    delta = max(deg.values(), default=0)
    at = {q: [None] * delta for q in deg}  # at[q][k] = neighbour reached via colour k

    for u, v in pairs:
        a = next(k for k in range(delta) if at[u][k] is None)  # colour free at u
        b = next(k for k in range(delta) if at[v][k] is None)  # colour free at v
        if a != b and at[v][a] is not None:
            # Swap colours a/b on the alternating path from v (it never reaches u).
            path, x, k = [], v, a
            while at[x][k] is not None:
                y = at[x][k]
                path.append((x, y, k))
                x, k = y, (b if k == a else a)
            for x, y, k in path:
                at[x][k] = at[y][k] = None
            for x, y, k in path:
                nk = b if k == a else a
                at[x][nk], at[y][nk] = y, x
        at[u][a], at[v][a] = v, u

    layers = [[] for _ in range(delta)]
    for u, v in pairs:
        layers[next(k for k in range(delta) if at[u][k] == v)].append((u, v))
    return [layer for layer in layers if layer]