package evolution_of_trust.agent;

import evolution_of_trust.match.Match;
import population.Individual;

public class Devil extends Agent {
    public Devil() {
        super("Devil");
    }

    @Override
    public Individual clone() {
        return new Devil();
    }

    @Override
    public int choice(int prevPreviousOpponentChoice, int previousOpponentChoice) {
        return Match.CHEAT;
    }
}
