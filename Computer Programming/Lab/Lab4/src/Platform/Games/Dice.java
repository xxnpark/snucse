package Platform.Games;

public class Dice {
    public int playGame() {
        int userDice = roll();
        int opponentDice = roll();

        System.out.println(userDice + " " + opponentDice);

        if (userDice > opponentDice) return 1;
        else if (userDice < opponentDice) return -1;
        else return 0;
    }

    private int roll() {
        return (int)(Math.random() * 100);
    }
}
