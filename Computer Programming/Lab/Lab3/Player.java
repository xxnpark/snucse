public class Player {
    String userID;
    int health = 50;

    Player(String userID) {
        this.userID = userID;
    }

    public void attack(Player opponent) {
        opponent.health -= (int)(Math.random() * 5 + 1);
        if (opponent.health < 0) opponent.health = 0;
    }

    public void heal() {
        health += (int)(Math.random() * 3 + 1);
        if (health > 50) health = 50;
    }

    public boolean alive() {
        return health > 0;
    }

    public char getTactic() {
        if (Math.random() > 0.3) return 'a';
        else return 'h';
    }

}