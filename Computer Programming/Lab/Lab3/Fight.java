public class Fight {
    int timeLimit = 100;
    int currRound = 0;
    Player p1, p2;

    Fight(Player p1, Player p2) {
        this.p1 = p1;
        this.p2 = p2;
    }

    public void proceed() {
        System.out.println("Round " + currRound);

        attackHeal();
        currRound++;

        System.out.printf("%s health: %d\n", p1.userID, p1.health);
        System.out.printf("%s health: %d\n", p2.userID, p2.health);
    }

    private void attackHeal() {
        if (p1.getTactic() == 'a') p1.attack(p2);
        else p1.heal();

        if (p2.getTactic() == 'a') p2.attack(p1);
        else p2.heal();
    }

    public boolean isFinished() {
        boolean p1Alive = p1.alive();
        boolean p2Alive = p2.alive();
        boolean limitReached = currRound > timeLimit;

        return !p1Alive || !p2Alive || limitReached;
    }

    public Player getWinner() {
        if (p1.health > p2.health) return p1;
        else return p2;
    }
}
