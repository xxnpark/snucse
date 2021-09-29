public class Main {
    public static void main(String[] args) {
        Player superman = new Player("Superman");
        Player batman = new Player("Batman");
        Fight fight = new Fight(superman, batman);

        while (!fight.isFinished()) {
            fight.proceed();
        }

        System.out.println(fight.getWinner().userID + " is the winner!");
    }
}