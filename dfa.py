
from typing import List, Dict

class DFA:
    def __init__(self, name, states, start_state, accept_states, transitions):
        self.name = name
        self.states = set(states)
        self.start_state = start_state
        self.accept_states = set(accept_states)
        self.transitions = dict(transitions)

    def run(self, inputs: List[str]) -> bool:
        state = self.start_state
        for sym in inputs:
            sym = sym.strip().upper()
            state = self.transitions.get((state, sym), self.transitions.get((state, "*"), state))
        return state in self.accept_states

def build_default_dfas() -> Dict[str, DFA]:
    # Rapid-SYN DFA
    rapid_syn = DFA(
        name="RapidSYN",
        states={"S0","S1","S2","S3","OK"},
        start_state="S0",
        accept_states={"S3"},
        transitions={
            ("S0","SYN"): "S1",
            ("S1","SYN"): "S2",
            ("S2","SYN"): "S3",
            ("S1","SYN-ACK"): "S1",
            ("S2","SYN-ACK"): "S2",
            ("S3","SYN-ACK"): "S3",
            ("S1","ACK"): "OK",
            ("S2","ACK"): "OK",
            ("S3","ACK"): "OK",
            ("OK","*"): "OK",
            ("S0","*"): "S0",
            ("S1","*"): "S1",
            ("S2","*"): "S2",
            ("S3","*"): "S3",
        }
    )

    # RST-Scan DFA
    rst_scan = DFA(
        name="RSTScan",
        states={"A","R1","R2"},
        start_state="A",
        accept_states={"R2"},
        transitions={
            ("A","RST"): "R1",
            ("A","*"): "A",
            ("R1","RST"): "R2",
            ("R1","*"): "A",
            ("R2","*"): "R2",
        }
    )

    return {"RapidSYN": rapid_syn, "RSTScan": rst_scan}

def dfa_check(events: List[str]) -> Dict[str, bool]:
    dfas = build_default_dfas()
    ev = [e.strip().upper() for e in events]
    return {name: dfa.run(ev) for name, dfa in dfas.items()}
