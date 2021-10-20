package bank;

public class Session {

    private String sessionKey;
    private Bank bank;
    private boolean valid;
    private int transLimit = 3;
    private int transCount = 0;

    Session(String sessionKey,Bank bank){
        this.sessionKey = sessionKey;
        this.bank = bank;
        valid = true;
    }

    public boolean deposit(int amount) {
        if (!valid) {
            return false;
        }
        bank.deposit(sessionKey, amount);
        if (++transCount >= transLimit) {
            expireSession();
        }
        return true;
    }

    public boolean withdraw(int amount) {
        if (!valid) {
            return false;
        }
        bank.withdraw(sessionKey, amount);
        if (++transCount >= transLimit) {
            expireSession();
        }
        return true;
    }

    public boolean transfer(String targetId, int amount) {
        if (!valid) {
            return false;
        }
        bank.transfer(sessionKey, targetId, amount);
        if (++transCount >= transLimit) {
            expireSession();
        }
        return true;
    }

    public void expireSession() {
        this.valid = false;
    }
}
