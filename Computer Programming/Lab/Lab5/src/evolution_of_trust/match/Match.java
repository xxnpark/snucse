package evolution_of_trust.match;

import evolution_of_trust.agent.Agent;

public class Match {
    public static int CHEAT = 0;

    public static int COOPERATE = 1;

    public static int UNDEFINED = -1;

    // Sets the value each player gets for all possible cases
    protected static int ruleMatrix[][][] = {
        {
            {0, 0}, // eg) A cheats, B cheats
            {3, -1} // eg) A cheats, B cooperates
        },
        {
            {-1, 3}, // eg) A cooperates, B cheats
            {2, 2} // eg) A cooperates, B cooperates
        }
    };

    Agent agentA, agentB;

    int previousChoiceA, prevPreviousChoiceA, previousChoiceB, prevPreviousChoiceB;

    public Match(Agent agentA, Agent agentB) {
        this.agentA = agentA;
        this.agentB = agentB;
        previousChoiceA = UNDEFINED;
        prevPreviousChoiceA = UNDEFINED;
        previousChoiceB = UNDEFINED;
        prevPreviousChoiceB = UNDEFINED;
    }

    public void playGame() {
        int choiceA = agentA.choice(prevPreviousChoiceB, previousChoiceB);
        int choiceB = agentB.choice(prevPreviousChoiceA, previousChoiceA);

        agentA.setScore(agentA.getScore() + ruleMatrix[choiceA][choiceB][0]);
        agentB.setScore(agentB.getScore() + ruleMatrix[choiceA][choiceB][1]);

        prevPreviousChoiceA = previousChoiceA;
        prevPreviousChoiceB = previousChoiceB;
        previousChoiceA = choiceA;
        previousChoiceB = choiceB;
    }

}
