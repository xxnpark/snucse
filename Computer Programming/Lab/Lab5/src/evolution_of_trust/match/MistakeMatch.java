package evolution_of_trust.match;

import evolution_of_trust.agent.Agent;

public class MistakeMatch extends Match {
    public MistakeMatch(Agent agentA, Agent agentB) {
        super(agentA, agentB);
    }

    @Override
    public void playGame() {
        int _choiceA = agentA.choice(prevPreviousChoiceA, previousChoiceB);
        int _choiceB = agentB.choice(prevPreviousChoiceB, previousChoiceA);

        int choiceA = Math.random() > 0.05 ? _choiceA : 1 - _choiceA;
        int choiceB = Math.random() > 0.05 ? _choiceB : 1 - _choiceB;

        agentA.setScore(agentA.getScore() + ruleMatrix[choiceA][choiceB][0]);
        agentB.setScore(agentB.getScore() + ruleMatrix[choiceA][choiceB][1]);

        prevPreviousChoiceA = previousChoiceA;
        prevPreviousChoiceB = previousChoiceB;
        previousChoiceA = choiceA;
        previousChoiceB = choiceB;

    }

}
