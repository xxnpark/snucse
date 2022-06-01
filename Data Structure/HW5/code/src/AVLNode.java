public class AVLNode<K extends Comparable<K>, V> { // 색인 검색을 위한 key 저장, 값은 linked list로 관리
    public K key;
    public Node<V> headValue = new Node<>(null);;
    int numValues = 0;
    public AVLNode<K, V> left, right;
    public int height;

    public AVLNode(K item, V value) {
        key = item;
        append(value);
        left = right = new AVLNode<>(null, null, null, null, 0);
        height = 1;
    }

    public AVLNode(K item, V value, AVLNode<K, V> leftChild, AVLNode<K, V> rightChild) {
        key = item;
        append(value);
        left = leftChild;
        right = rightChild;
        height = 1;
    }

    public AVLNode(K item, V value, AVLNode<K, V> leftChild, AVLNode<K, V> rightChild, int h) {
        key = item;
        append(value);
        left = leftChild;
        right = rightChild;
        height = h;
    }

    public boolean isEmpty() {
        return headValue.getNext() == null;
    }

    public int size() {
        return numValues;
    }

    public V first() {
        return headValue.getNext().getItem();
    }

    public void append(V item) {
        Node<V> prevNode = headValue;
        while (prevNode.getNext() != null) {
            prevNode = prevNode.getNext();
        }
        prevNode.setNext(new Node<>(item, null));
        numValues++;
    }

    public void remove(V item) {
        Node<V> prev;
        Node<V> last = headValue;

        while (last.getNext() != null) {
            prev = last;
            last = last.getNext();

            if (last.getItem().equals(item)) {
                prev.removeNext();
                numValues--;
            }
        }
    }

    public void removeAll() {
        headValue.setNext(null);
    }
}
