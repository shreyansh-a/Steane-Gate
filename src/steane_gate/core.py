import numpy as np

from .util import _Circuit, get_schedule


def _xor(*sets):
    """XOR of measurement-index sets; None means the value is random."""
    out = set()
    for s in sets:
        if s is None:
            return None
        out ^= s
    return out


class Steane_Circuit(_Circuit):
    """
    Circuit made of [[7,1,3]] Steane code blocks with transversal logical gates.

    Besides the gates, the class tracks, for every block and every check, which
    earlier measurement results its next outcome should equal. Syndrome rounds
    and logical measurements use that to add Stim DETECTORs.

    Detector coordinates are (block, check, basis) with basis 0 = X, 1 = Z.
    """

    # Parity checks of the [7,4,3] Hamming code. They are used for both X and Z
    # checks, so every pair of rows must overlap on an even number of qubits.
    STABILIZER_TABLE = np.array([[1, 0, 1, 0, 1, 0, 1],
                                 [0, 1, 1, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 1, 1, 1]])

    def __init__(self, blocks: int, ancillas: bool = False):
        """
        Builds a circuit with `blocks` Steane code blocks. If `ancillas` is
        True, every block gets 6 ancillas: 3 for its X checks and 3 for its Z checks.
        """
        per_block = 13 if ancillas else 7
        super().__init__(per_block * blocks)
        self.num_blocks = blocks
        offsets = np.arange(blocks)[:, None] * per_block
        self.data = offsets + np.arange(7)
        self.ancillas = offsets + np.arange(7, 13) if ancillas else None

        # expected[basis][block][check]: set of measurement indices whose XOR
        # predicts the next outcome of that check, or None if it is random.
        # Every qubit starts in |0>: Z checks are +1, X checks are random.
        self.expected = {
            "X": [[None] * 3 for _ in range(blocks)],
            "Z": [[set() for _ in range(3)] for _ in range(blocks)],
        }

        # A CNOT order for a single block, copied to every block.
        # NOTE: with one bare ancilla per weight-4 check, no order prevents hook
        # errors, so this extraction is not fully fault-tolerant at distance 3.
        template = [(7 + r, int(q)) for r, row in enumerate(self.STABILIZER_TABLE)
                    for q in np.flatnonzero(row)]
        self._layers = [[(a - 7, q) for a, q in layer] for layer in get_schedule(template)]

    # ------------------------------------------------------------------ helpers
    def _blocks(self, blocks):
        blocks = [int(b) for b in np.atleast_1d(blocks)]
        if len(blocks) != len(set(blocks)):
            raise ValueError("A block appears more than once in one call")
        return blocks

    def _data_of(self, blocks):
        return np.concatenate([self.data[b] for b in blocks]) if blocks else []

    def _set_expected(self, basis, b, values):
        self.expected[basis][b] = [None if v is None else set(v) for v in values]

    # ------------------------------------------------------------- logical ops
    def op(self, name, blocks):
        """
        Applies a transversal logical gate to the given blocks.
          'X', 'Z' : logical Pauli
          'H'      : logical Hadamard
          'S', 'ST': logical S and S-dagger (physical S_DAG and S, respectively)
          'R'      : physical reset to |0>^7 (logical |0> once an X-syndrome round has run)
          'RX'     : physical reset to |+>^7 (logical |+> once a Z-syndrome round has run)
        """
        blocks = self._blocks(blocks)
        q = self._data_of(blocks)
        if name in ("X", "Z"):
            super().op(name, q)
        elif name == "H":
            super().op("H", q)
            for b in blocks:
                x, z = self.expected["X"][b], self.expected["Z"][b]
                self.expected["X"][b], self.expected["Z"][b] = z, x
        elif name in ("S", "ST"):
            super().op("S_DAG" if name == "S" else "S", q)
            # S maps each X check to X*Z on the same support (sign +1), so the
            # next X outcome equals old X XOR old Z; Z checks are unchanged.
            for b in blocks:
                self._set_expected("X", b, [_xor(x, z) for x, z in
                                            zip(self.expected["X"][b], self.expected["Z"][b])])
        elif name in ("R", "RX"):
            super().op(name, q)
            fixed, rand = ("Z", "X") if name == "R" else ("X", "Z")
            for b in blocks:
                self._set_expected(fixed, b, [set()] * 3)
                self._set_expected(rand, b, [None] * 3)
        else:
            raise ValueError(f"Unsupported logical gate {name!r}")

    def cx(self, pairs):
        """
        Transversal logical CNOT on (control_block, target_block) pairs. Each
        block may appear only once per call. Stabilizers propagate as
        X_c -> X_c X_t and Z_t -> Z_c Z_t, so the next X outcome on the control
        is old X_c XOR old X_t, and the next Z outcome on the target is
        old Z_t XOR old Z_c.
        """
        pairs = [(int(c), int(t)) for c, t in pairs]
        self._blocks([b for p in pairs for b in p])
        super().cx([(int(qc), int(qt)) for c, t in pairs
                    for qc, qt in zip(self.data[c], self.data[t])])
        for c, t in pairs:
            ex, ez = self.expected["X"], self.expected["Z"]
            ex[c] = [_xor(a, b) for a, b in zip(ex[c], ex[t])]
            ez[t] = [_xor(a, b) for a, b in zip(ez[t], ez[c])]

    # ------------------------------------------------------- error correction
    def measure_syndrome(self, basis, blocks=None):
        """
        One round of X or Z syndrome extraction on `blocks` (default: all).
        Adds a detector for every check whose outcome is predictable and returns
        the measurement record indices with shape (len(blocks), 3).
        """
        if self.ancillas is None:
            raise ValueError("Circuit has no ancillas")
        if basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")
        blocks = self._blocks(range(self.num_blocks) if blocks is None else blocks)
        anc = [self.ancillas[b][0:3] if basis == "X" else self.ancillas[b][3:6]
               for b in blocks]
        all_anc = np.concatenate(anc)

        super().op("RX" if basis == "X" else "R", all_anc)
        super().tick()
        for layer in self._layers:
            if basis == "X":   # X check: ancilla in |+> controls CNOTs onto data
                super().cx([(a[r], self.data[b][q]) for b, a in zip(blocks, anc) for r, q in layer])
            else:              # Z check: data controls CNOTs onto ancilla in |0>
                super().cx([(self.data[b][q], a[r]) for b, a in zip(blocks, anc) for r, q in layer])
            super().tick()
        recs = super().op("MX" if basis == "X" else "M", all_anc)

        out = np.array(recs).reshape(len(blocks), 3)
        for i, b in enumerate(blocks):
            for k in range(3):
                prev = self.expected[basis][b][k]
                if prev is not None:
                    self.detector(prev | {int(out[i, k])}, (b, k, 0 if basis == "X" else 1))
                self.expected[basis][b][k] = {int(out[i, k])}
        super().tick()
        return out

    def measure(self, blocks, basis="Z", observable=None):
        """
        Destructive logical measurement of `blocks` in the X or Z basis. Adds
        detectors for the matching checks (rebuilt from the data results).
        Returns one list of 7 record indices per block; the logical value is
        their XOR. With `observable=i`, that XOR is added to logical
        observable i (one observable per block, starting at i).
        """
        if basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")
        blocks = self._blocks(blocks)
        recs = super().op("MX" if basis == "X" else "M", self._data_of(blocks))
        out = [recs[7 * i:7 * i + 7] for i in range(len(blocks))]
        for i, b in enumerate(blocks):
            for k, row in enumerate(self.STABILIZER_TABLE):
                prev = self.expected[basis][b][k]
                if prev is not None:
                    parity = {out[i][q] for q in np.flatnonzero(row)}
                    self.detector(prev ^ parity, (b, k, 0 if basis == "X" else 1))
            # The block's qubits are no longer in a code state.
            self._set_expected("X", b, [None] * 3)
            self._set_expected("Z", b, [None] * 3)
            if observable is not None:
                self.observable(out[i], observable + i)
        return out


class LookupDecoder:
    """
    Most-likely-observable decoder built from the circuit's detector error model.

    Enumerates every combination of at most `max_weight` (1 to 3) error
    mechanisms and, for each syndrome, stores the most probable set of
    observable flips. All observables are decoded jointly.

    `postselect` lists detector indices used to discard shots rather than to
    decode. Combinations that fire any of them are excluded, but combinations
    whose effects on them cancel still count, since those shots are kept.
    Use `accepted` to find the kept shots.
    """

    def __init__(self, circuit, max_weight=3, postselect=()):
        if max_weight not in (1, 2, 3):
            raise ValueError("max_weight must be 1, 2 or 3")
        num_detectors = circuit.num_detectors
        post_ids = sorted({int(d) for d in postselect})
        if post_ids and not 0 <= post_ids[0] <= post_ids[-1] < num_detectors:
            raise ValueError("postselect contains an index that is not a detector")
        post_set = set(post_ids)
        keep_ids = [d for d in range(num_detectors) if d not in post_set]
        if max(len(keep_ids), len(post_ids), circuit.num_observables) > 62:
            raise ValueError("at most 62 decoded detectors, 62 postselected "
                             "detectors and 62 observables fit an int64 lookup key")
        self.num_detectors = num_detectors
        self.num_observables = circuit.num_observables
        self.post = np.array(post_ids, dtype=int)
        self.keep = np.array(keep_ids, dtype=int)
        keep_col = {d: i for i, d in enumerate(keep_ids)}
        post_col = {d: i for i, d in enumerate(post_ids)}

        merged = {}  # (syndrome bits, postselection bits, observable bits) -> probability
        for inst in circuit.detector_error_model(decompose_errors=False).flattened():
            if inst.type != "error":
                continue
            syn, post, obs = 0, 0, 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    if t.val in post_col:
                        post ^= 1 << post_col[t.val]
                    else:
                        syn ^= 1 << keep_col[t.val]
                elif t.is_logical_observable_id():
                    obs ^= 1 << t.val
            if syn or post or obs:
                p, q = merged.get((syn, post, obs), 0.0), inst.args_copy()[0]
                merged[(syn, post, obs)] = p * (1 - q) + q * (1 - p)

        syn, post, obs = (np.array([k[n] for k in merged], dtype=np.int64) for n in range(3))
        odds = np.array([v / (1 - v) for v in merged.values()], dtype=float)

        # The no-error case: empty syndrome, no flips, weight 1.
        chunks = [(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.ones(1))]

        def add(s, p, o, w):
            survives = p == 0
            chunks.append((s[survives], o[survives], w[survives]))

        add(syn, post, obs, odds)
        if max_weight >= 2:
            i, j = np.triu_indices(len(syn), k=1)
            ps, pp, po, pw = syn[i] ^ syn[j], post[i] ^ post[j], obs[i] ^ obs[j], odds[i] * odds[j]
            add(ps, pp, po, pw)
        if max_weight >= 3:
            # pairs are sorted by their first index, so pairs (j,k) with j > a start at start[a+1]
            start = np.searchsorted(i, np.arange(len(syn) + 1))
            for a in range(len(syn) - 2):
                sl = slice(start[a + 1], None)
                add(syn[a] ^ ps[sl], post[a] ^ pp[sl], obs[a] ^ po[sl], odds[a] * pw[sl])
        s_all, o_all, w_all = map(np.concatenate, zip(*chunks))

        # Total probability of each (syndrome, observable flips) pair.
        order = np.lexsort((o_all, s_all))
        s_all, o_all, w_all = s_all[order], o_all[order], w_all[order]
        first = np.flatnonzero(np.r_[True, (s_all[1:] != s_all[:-1]) | (o_all[1:] != o_all[:-1])])
        mass = np.add.reduceat(w_all, first)
        s_all, o_all = s_all[first], o_all[first]

        # For each syndrome keep the largest mass; ties go to the smaller observable mask.
        order = np.lexsort((-o_all, mass, s_all))
        s_all, o_all = s_all[order], o_all[order]
        last = np.r_[s_all[1:] != s_all[:-1], True]
        self.keys = s_all[last]
        self.masks = o_all[last]

    def _check(self, dets):
        dets = np.asarray(dets, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"dets must have shape (shots, {self.num_detectors})")
        return dets

    def accepted(self, dets):
        """Boolean array of shape (shots,): True where no postselected detector fired."""
        return ~self._check(dets)[:, self.post].any(axis=1)

    def decode(self, dets):
        """
        Predicted observable flips, as a boolean array of shape
        (shots, num_observables). Unseen syndromes get no correction.
        """
        bits = self._check(dets)[:, self.keep].astype(np.int64)
        s = bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))
        pos = np.clip(np.searchsorted(self.keys, s), 0, len(self.keys) - 1)
        masks = np.where(self.keys[pos] == s, self.masks[pos], 0)
        return (masks[:, None] >> np.arange(self.num_observables, dtype=np.int64)) & 1 == 1
