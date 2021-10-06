package evolution_of_trust.agent;

import population.Individual;

abstract public class Agent extends Individual {
    private int score;

    private String name;

    public Agent(String name) {
        score = 0;
        this.name = name;
    }

    public int sortKey() {
        return getScore();
    }

    @Override
    public String toString() {
        return name + ": " + getScore();
    }

    public int getScore() {
        return score;
    }

    public void setScore(int newScore) {
        score = newScore;
    }

    abstract public int choice(int prevPreviousOpponentChoice, int previousOpponentChoice);
}
