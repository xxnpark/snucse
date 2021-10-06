package evolution_of_trust.agent;

import evolution_of_trust.match.Match;
import population.Individual;

import evolution_of_trust.match.Match;
import population.Individual;

public class Copykitten extends Agent{
    public Copykitten() {
        super("Copykitten");
    }

    @Override
    public Individual clone() {
        return new Copykitten();
    }

    @Override
    public int choice(int prevPreviousOpponentChoice, int previousOpponentChoice) {
        if (prevPreviousOpponentChoice == Match.CHEAT && previousOpponentChoice == Match.CHEAT) {
            return Match.CHEAT;
        } else {
            return Match.COOPERATE;
        }
    }
}
