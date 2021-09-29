package Platform.Games;

import java.util.Scanner;

public class RockPaperScissors {
    private String[] choices = {"rock", "paper", "scissor"};

    public int playGame() {
        Scanner sc = new Scanner(System.in);
        String userInput = sc.next();

        int userChoice = inputToInt(userInput); // 0 rock, 1 paper, 2 scissor
        int opponentChoice = (int)(Math.random() * 3); // 0 rock, 1 paper, 2 scissor

        System.out.println(userInput + " " + choices[opponentChoice]);

        if (userChoice == -1) return -1;
        else return compare(userChoice, opponentChoice);
    }

    private int inputToInt(String input) {
        for (int i = 0; i < choices.length; i++) {
            if (choices[i].equals(input)) return i;
        }
        return -1;
    }

    private int compare(int userChoice, int opponentChoice) {
        if (userChoice == 0 && opponentChoice == 2) return 1;
        else if (userChoice == 2 && opponentChoice == 0) return -1;
        else if (userChoice > opponentChoice) return 1;
        else if (userChoice < opponentChoice) return -1;
        else return 0;
    }
}
