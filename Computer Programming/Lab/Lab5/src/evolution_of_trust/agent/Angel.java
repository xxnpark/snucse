package evolution_of_trust.agent;

import evolution_of_trust.match.Match;
import population.Individual;

public class Angel extends Agent {
    public Angel() {
        super("Angel");
    }

    @Override
    public Individual clone() {
        return new Angel();
    }

    @Override
    public int choice(int prevPreviousOpponentChoice, int previousOpponentChoice) {
        return Match.COOPERATE;
    }
}
