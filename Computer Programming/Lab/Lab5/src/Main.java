import evolution_of_trust.agent.Agent;
import evolution_of_trust.environment.Tournament;
import evolution_of_trust.match.Match;
import evolution_of_trust.agent.Copycat;
import evolution_of_trust.agent.Devil;

public class Main {
    public static void main(String[] args) {
        Tournament tournament = new Tournament();

        for (int i = 0; i < 10; i++) {
            tournament.resetAgents();
            tournament.playAllGames(10);
            tournament.describe();
            tournament.evolvePopulation();
        }
    }
}
