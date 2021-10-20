package hand.market;

import hand.agent.Buyer;
import hand.agent.Seller;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

class Pair<K,V> {
    public K key;
    public V value;
    Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }
}

public class Market {
    public ArrayList<Buyer> buyers;
    public ArrayList<Seller> sellers;

    public Market(int nb, ArrayList<Double> fb, int ns, ArrayList<Double> fs) {
        buyers = createBuyers(nb, fb);
        sellers = createSellers(ns, fs);
    }

    private double polynomial(double d, ArrayList<Double> f) {
        double ret = 0;
        for (int i = 0; i < f.size(); i++) {
            ret += f.get(i) * Math.pow(d, i);
        }
        return ret;
    }
    
    private ArrayList<Buyer> createBuyers(int n, ArrayList<Double> f) {
        ArrayList<Buyer> buyers = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            buyers.add(new Buyer(polynomial(i/(double)n, f)));
        }
        return buyers;
    }

    private ArrayList<Seller> createSellers(int n, ArrayList<Double> f) {
        ArrayList<Seller> sellers = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            sellers.add(new Seller(polynomial(i/(double)n, f)));
        }
        return sellers;
    }

    private ArrayList<Pair<Seller, Buyer>> matchedPairs(int day, int round) {
        ArrayList<Seller> shuffledSellers = new ArrayList<>(sellers);
        ArrayList<Buyer> shuffledBuyers = new ArrayList<>(buyers);
        Collections.shuffle(shuffledSellers, new Random(71 * day + 43 * round + 7));
        Collections.shuffle(shuffledBuyers, new Random(67 * day + 29 * round + 11));
        ArrayList<Pair<Seller, Buyer>> pairs = new ArrayList<>();
        for (int i = 0; i < shuffledBuyers.size(); i++) {
            if (i < shuffledSellers.size()) {
                pairs.add(new Pair<>(shuffledSellers.get(i), shuffledBuyers.get(i)));
            }
        }
        return pairs;
    }

    public double simulate() {
        double sum = 0;
        int count = 0;

        for (int day = 1; day <= 3000; day++) { // do not change this line
            for (int round = 1; round <= 5; round++) { // do not change this line
                ArrayList<Pair<Seller, Buyer>> pairs = matchedPairs(day, round); // do not change this line
                for (Pair<Seller, Buyer> pair: pairs) {
                    Seller seller = pair.key;
                    Buyer buyer = pair.value;
                    if (seller.willTransact(buyer.getExpectedPrice()) && buyer.willTransact(seller.getExpectedPrice())) {
                        seller.makeTransaction();
                        buyer.makeTransaction();
                        if (day == 3000) {
                            sum += seller.getExpectedPrice();
                            count ++;
                        }
                    }
                }
            }

            for (Seller seller: sellers) {
                seller.reflect();
            }
            for (Buyer buyer: buyers) {
                buyer.reflect();
            }
        }

        return sum / count;
    }
}

