package bank;

import bank.event.*;

import java.util.Arrays;

class BankAccount {
    private int numEvents = 0;
    private Event[] events = new Event[maxEvents];
    final static int maxEvents = 100;

    private String id;
    private String password;
    private int balance;


    BankAccount(String id, String password, int balance) {
        this.id = id;
        this.password = password;
        this.balance = balance;
    }


    boolean authenticate(String password) {
        return this.password.equals(password);
    }

    void deposit(int amount) {
        this.balance += amount;
        events[numEvents++] = new DepositEvent();
    }

    boolean withdraw(int amount) {
        if (this.balance >= amount) {
            this.balance -= amount;
            events[numEvents++] = new WithdrawEvent();
            return true;
        }
        return false;
    }

    void receive(int amount) {
        this.balance += amount;
        events[numEvents++] = new ReceiveEvent();
    }

    boolean send(int amount) {
        if (this.balance >= amount) {
            this.balance -= amount;
            events[numEvents++] = new SendEvent();
            return true;
        }
        return false;
    }

    Event[] getEvents() {
        return Arrays.copyOfRange(events, 0, numEvents);
    }

    public int getBalance() {
        return balance;
    }
}
