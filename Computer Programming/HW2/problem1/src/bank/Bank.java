package bank;

import bank.event.*;
import security.*;
import security.key.*;

import java.util.HashMap;

public class Bank {
    private int numAccounts = 0;
    final static int maxAccounts = 100;
    private BankAccount[] accounts = new BankAccount[maxAccounts];
    private String[] ids = new String[maxAccounts];

    public void createAccount(String id, String password) {
        createAccount(id, password, 0);
    }

    public void createAccount(String id, String password, int initBalance) {
        if (find(id) == null) {
            accounts[numAccounts] = new BankAccount(id, password, initBalance);
            ids[numAccounts++] = id;
        }
    }

    public boolean deposit(String id, String password, int amount) {
        BankAccount account = authenticate(id, password);
        if (account == null) {
            return false;
        }
        account.deposit(amount);
        return true;
    }

    public boolean withdraw(String id, String password, int amount) {
        BankAccount account = authenticate(id, password);
        if (account == null) {
            return false;
        }
        return account.withdraw(amount);
    }

    public boolean transfer(String sourceId, String password, String targetId, int amount) {
        BankAccount sourceAccount = authenticate(sourceId, password);
        BankAccount targetAccount = find(targetId);
        if (sourceAccount == null || targetAccount == null || !sourceAccount.send(amount)) {
            return false;
        }
        targetAccount.receive(amount);
        return true;
    }

    public Event[] getEvents(String id, String password) {
        BankAccount account = authenticate(id, password);
        if (account == null) {
            return null;
        }
        return account.getEvents();
    }

    public int getBalance(String id, String password) {
        BankAccount account = authenticate(id, password);
        if (account == null) {
            return -1;
        }
        return account.getBalance();
    }

    private static String randomUniqueStringGen(){
        return Encryptor.randomUniqueStringGen();
    }

    private BankAccount find(String id) {
        for (int i = 0; i < numAccounts; i++) {
            if(ids[i].equals(id)){return accounts[i];};
        }
        return null;
    }

    private BankAccount authenticate(String id, String password) {
        BankAccount account = find(id);
        if (account != null && account.authenticate(password)) {
            return account;
        }
        return null;
    }

    final static int maxSessionKey = 100;
    int numSessionKey = 0;
    String[] sessionKeyArr = new String[maxSessionKey];
    BankAccount[] bankAccountmap = new BankAccount[maxSessionKey];
    String generateSessionKey(String id, String password){
        BankAccount account = find(id);
        if(account == null || !account.authenticate(password)){
            return null;
        }
        String sessionkey = randomUniqueStringGen();
        sessionKeyArr[numSessionKey] = sessionkey;
        bankAccountmap[numSessionKey] = account;
        numSessionKey += 1;
        return sessionkey;
    }
    BankAccount getAccount(String sessionkey){
        for(int i = 0 ;i < numSessionKey; i++){
            if(sessionKeyArr[i] != null && sessionKeyArr[i].equals(sessionkey)){
                return bankAccountmap[i];
            }
        }
        return null;
    }

    boolean deposit(String sessionkey, int amount) {
        BankAccount account = getAccount(sessionkey);
        if (account == null) {
            return false;
        }
        account.deposit(amount);
        return true;
    }

    boolean withdraw(String sessionkey, int amount) {
        BankAccount account = getAccount(sessionkey);
        if (account == null) {
            return false;
        }
        return account.withdraw(amount);
    }

    boolean transfer(String sessionkey, String targetId, int amount) {
        BankAccount sourceAccount = getAccount(sessionkey);
        BankAccount targetAccount = find(targetId);
        if (sourceAccount == null || targetAccount == null || !sourceAccount.send(amount)) {
            return false;
        }
        targetAccount.receive(amount);
        return true;
    }

    private BankSecretKey secretKey;
    public BankPublicKey getPublicKey(){
        BankKeyPair keypair = Encryptor.publicKeyGen(); // generates two keys : BankPublicKey, BankSecretKey
        secretKey = keypair.deckey; // stores BankSecretKey internally
        return keypair.enckey;
    }

    int maxHandshakes = 10000;
    int numSymmetrickeys = 0;
    BankSymmetricKey[] bankSymmetricKeys = new BankSymmetricKey[maxHandshakes];
    String[] AppIds = new String[maxHandshakes];

    public int getAppIdIndex(String AppId){
        for(int i=0; i<numSymmetrickeys; i++){
            if(AppIds[i].equals(AppId)){
                return i;
            }
        }
        return -1;
    }

    private HashMap<String, BankSymmetricKey> keyMap = new HashMap<>();
    public void fetchSymKey(Encrypted<BankSymmetricKey> encryptedKey, String AppId){
        keyMap.put(AppId, encryptedKey.decrypt(secretKey));
    }

    public Encrypted<Boolean> processRequest(Encrypted<Message> messageEnc, String AppId) {
        BankSymmetricKey symmetricKey = keyMap.get(AppId);
        if (symmetricKey == null || messageEnc == null) {
            return null;
        }
        Message message = messageEnc.decrypt(symmetricKey);
        if (message == null) {
            return null;
        }

        boolean result;
        if (message.getRequestType().equals("deposit")) {
            result = deposit(message.getId(), message.getPassword(), message.getAmount());
        } else if (message.getRequestType().equals("withdraw")) {
            result = withdraw(message.getId(), message.getPassword(), message.getAmount());
        } else {
            return null;
        }
        return new Encrypted<>(result, symmetricKey);
    }
}