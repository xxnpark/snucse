public class AVLTree<K extends Comparable<K>, V> {
    public AVLNode<K, V> root;
    final AVLNode<K, V> NIL = new AVLNode<>(null, null, null, null, 0);

    public AVLTree() {
        root = NIL;
    }

    // Search 검색

    public AVLNode<K, V> search(K key) {
        return searchItem(root, key);
    }

    private AVLNode<K, V> searchItem(AVLNode<K, V> tNode, K key) {
        if (tNode == NIL){
            return NIL;
        } else if (key.compareTo(tNode.key) == 0) {
            return tNode;
        } else if (key.compareTo(tNode.key) < 0) {
            return searchItem(tNode.left, key);
        } else {
            return searchItem(tNode.right, key);
        }
    }

    // Insertion 삽입

    public void insert(K key, V value) {
        root = insertItem(root, key, value);
    }

    public AVLNode<K, V> insertItem(AVLNode<K, V> tNode, K key, V value) {
        if (tNode == NIL) {
            tNode = new AVLNode<>(key, value, NIL, NIL);
        } else if (key.compareTo(tNode.key) < 0) {
            tNode.left = insertItem(tNode.left, key, value);
            tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
            int type = needBalance(tNode);
            if (type != NO_NEED) {
                tNode = balanceAVL(tNode, type);
            }
        } else if (key.compareTo(tNode.key) > 0) {
            tNode.right = insertItem(tNode.right, key, value);
            tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
            int type = needBalance(tNode);
            if (type != NO_NEED) {
                tNode = balanceAVL(tNode, type);
            }
        } else {
            tNode.append(value);
        }

        return tNode;
    }

    // Deletion 삭제

    public void delete(K key) {
        root = deleteItem(root, key);
    }

    private AVLNode<K, V> deleteItem(AVLNode<K, V> tNode, K key) {
        if (tNode == NIL) {
            return NIL;
        } else {
            if (key.compareTo(tNode.key) == 0) {
                tNode = deleteNode(tNode);
            } else if (key.compareTo(tNode.key) < 0) {
                tNode.left = deleteItem(tNode.left, key);
                tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
                int type = needBalance(tNode);
                if (type != NO_NEED) {
                    tNode = balanceAVL(tNode, type);
                }
            } else {
                tNode.right = deleteItem(tNode.right, key);
                tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
                int type = needBalance(tNode);
                if (type != NO_NEED) {
                    tNode = balanceAVL(tNode, type);
                }
            }

            return tNode;
        }
    }

    private AVLNode<K, V> deleteNode(AVLNode<K, V> tNode) {
        if (tNode.left == NIL && tNode.right == NIL) {
            return NIL;
        } else if (tNode.left == NIL) {
            return tNode.right;
        } else if (tNode.right == NIL) {
            return tNode.left;
        } else {
            returnPair rPair = deleteMinItem(tNode.right);
            tNode.key = rPair.item;
            tNode.right = rPair.node;
            tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
            int type = needBalance(tNode);
            if (type != NO_NEED) {
                tNode = balanceAVL(tNode, type);
            }
            return tNode;
        }
    }

    private returnPair deleteMinItem(AVLNode<K, V> tNode) {
        int type;
        if (tNode.left == NIL) {
            return new returnPair(tNode.key, tNode.right);
        } else {
            returnPair rPair = deleteMinItem(tNode.left);
            tNode.left = rPair.node;
            tNode.height = 1 + Math.max(tNode.right.height, tNode.left.height);
            type = needBalance(tNode);
            if (type != NO_NEED) {
                tNode=  balanceAVL(tNode, type);
            }
            rPair.node = tNode;
            return rPair;
        }
    }

    // Balancing 균형

    private final int LL = 1, LR = 2, RR = 3, RL = 4, NO_NEED = 0, ILLEGAL = -1;

    private int needBalance(AVLNode<K, V> tNode) {
        int type = ILLEGAL;

        if (tNode.left.height + 1 < tNode.right.height) {
            if (tNode.right.left.height <= tNode.right.right.height) {
                type = RR;
            } else {
                type = RL;
            }
        } else if (tNode.left.height > tNode.right.height + 1) {
            if (tNode.left.left.height >= tNode.left.right.height) {
                type = LL;
            } else {
                type = LR;
            }
        } else {
            type = NO_NEED;
        }

        return type;
    }

    private AVLNode<K, V> balanceAVL(AVLNode<K, V> tNode, int type) {
        AVLNode<K, V> returnNode = NIL;

        switch (type) {
            case LL:
                returnNode = rightRotate(tNode);
                break;
            case LR:
                tNode.left = leftRotate(tNode.left);
                returnNode = rightRotate(tNode);
                break;
            case RR:
                returnNode = leftRotate(tNode);
                break;
            case RL:
                tNode.right = rightRotate(tNode.right);
                returnNode = leftRotate(tNode);
                break;
            default:
                System.out.println("Impossible type! Should be one of LL, LR, RR< RL");
                break;
        }

        return returnNode;
    }

    private AVLNode<K, V> leftRotate(AVLNode<K, V> tNode) {
        AVLNode<K, V> RChild = tNode.right;
        if (RChild == NIL) {
            System.out.println(tNode.key + "'s RChild shouldn't be NIL!");
        }
        AVLNode<K, V> RLChild = RChild.left;
        RChild.left = tNode;
        tNode.right = RLChild;
        tNode.height = 1 + Math.max(tNode.left.height, tNode.right.height);
        RChild.height = 1 + Math.max(RChild.left.height, RChild.right.height);
        return RChild;
    }

    private AVLNode<K, V> rightRotate(AVLNode<K, V> tNode) {
        AVLNode<K, V> LChild = tNode.left;
        if (LChild == NIL) {
            System.out.println(tNode.key + "'s LChild shouldn't be NIL!");
        }
        AVLNode<K, V> LRChild = LChild.right;
        LChild.right = tNode;
        tNode.left = LRChild;
        tNode.height = 1 + Math.max(tNode.left.height, tNode.right.height);
        LChild.height = 1 + Math.max(LChild.left.height, LChild.right.height);
        return LChild;
    }

    // 기타

    public boolean isEmpty() {
        return root == NIL;
    }

    public void clear() {
        root = NIL;
    }

    public String printPreOrder(AVLNode<K, V> tNode) {
        StringBuilder stringBuilder = new StringBuilder();
        if (tNode == NIL) {
            return "";
        } else {
            stringBuilder.append(" ").append(tNode.key).append(printPreOrder(tNode.left)).append(printPreOrder(tNode.right));
        }
        return stringBuilder.toString();
    }

    public String printPreOrderRoot() {
        return printPreOrder(root);
    }

    // Pair 클래스

    private class returnPair {
        private K item;
        private AVLNode<K, V> node;
        private returnPair(K item, AVLNode<K, V> node) {
            this.item = item;
            this.node = node;
        }
    }
}
