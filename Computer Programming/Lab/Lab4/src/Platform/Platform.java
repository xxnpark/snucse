package Platform;

import Platform.Games.Dice;
import Platform.Games.RockPaperScissors;
import java.util.Scanner;

public class Platform {
    private int rounds = 1;
    private boolean roundSet = false;

    public float run() {
        Scanner sc = new Scanner(System.in);
        int game = sc.nextInt();
        int wins = 0;

        setRounds((int)(Math.random() * 6 + 5));

        if (game == 0) wins = runDice();
        else wins = runRPS(); // game == 1

        return wins / (float)rounds;
    }

    public void setRounds(int num) {
        if (!roundSet) {
            roundSet = true;
            rounds = num;
        }
    }

    private int runDice() {
        Dice dice = new Dice();
        int sum = 0;
        for (int i = 0; i < rounds; i++) {
            int status = dice.playGame();
            if(status == 1) sum += 1;
        }
        return sum;
    }

    private int runRPS() {
        RockPaperScissors rps = new RockPaperScissors();
        int sum = 0;
        for (int i = 0; i < rounds; i++) {
            int status = rps.playGame();
            if(status == 1) sum += 1;
        }
        return sum;
    }
}
