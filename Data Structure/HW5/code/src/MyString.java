public class MyString implements Comparable<MyString>{
    private String str;

    public MyString(String str) {
        this.str = str;
    }

    public String getStr() {
        return str;
    }

    @Override
    public int compareTo(MyString myString) {
        return str.compareTo(myString.str);
    }

    @Override
    public int hashCode() {
        int sum = 0;
        for (int i = 0; i < str.length(); i++) {
            sum += str.charAt(i);
        }
        return sum % 100;
    }

    @Override
    public String toString() {
        return str;
    }
}
