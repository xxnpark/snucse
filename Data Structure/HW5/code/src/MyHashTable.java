import java.util.Hashtable;

public class MyHashTable<K extends Comparable<K>, V> {
    private AVLTree<K, V>[] table;
    int numItems;

    public MyHashTable(int n) {
        table = (AVLTree<K, V>[])new AVLTree[n];
        for (int i = 0; i < n; i++) {
            table[i] = new AVLTree<>();
        }
        numItems = 0;
    }

    public void insert(K key, V value) {
        table[key.hashCode()].insert(key, value);
        numItems++;
    }

    public AVLTree<K, V> search(int slot) {
        return table[slot];
    }

}
