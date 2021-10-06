package evolution_of_trust.agent;

import evolution_of_trust.match.Match;
import population.Individual;

public class Copycat extends Agent {
    public Copycat() {
        super("Copycat");
    }

    @Override
    public Individual clone() {
        return new Copycat();
    }

    @Override
    public int choice(int prevPreviousOpponentChoice, int previousOpponentChoice) {
        if (previousOpponentChoice == Match.UNDEFINED) {
            return Match.COOPERATE;
        } else {
            return previousOpponentChoice;
        }
    }
}
