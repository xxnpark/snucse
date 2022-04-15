import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
  
  
public class BigInteger {
    public static final String QUIT_COMMAND = "quit";
    public static final String MSG_INVALID_INPUT = "입력이 잘못되었습니다.";

    public static final Pattern EXPRESSION_PATTERN = Pattern.compile("[+\\-]?[0-9]+");

    private int[] num; // reversed
    private boolean sign;

    public BigInteger(int i) {
        sign = i >= 0;
        num = new int[]{Math.abs(i)};
    }

    // always positive and reversed input
    public BigInteger(int[] num1) {
        sign = true;

        int countFirstZeros = 0;
        for (int i = num1.length - 1; i >= 0; i--) {
            if (num1[i] == 0) countFirstZeros++;
            else break;
        }

        if (num1.length == countFirstZeros) {
            num = new int[]{0};
        } else {
            num = new int[num1.length - countFirstZeros];
            for (int i = 0; i < num1.length - countFirstZeros; i++) {
                num[i] = num1[i];
            }
        }
    }

    public BigInteger(String s) {
        int length;

        if (s.charAt(0) == '-') {
            sign = false;
            length = s.length() - 1;
        } else if (s.charAt(0) == '+') {
            sign = true;
            length = s.length() - 1;
        } else {
            sign = true;
            length = s.length();
        }

        num = new int[length];
        String[] numStr = s.split("");
        for (int i = 0; i < length; i++) {
            num[i] = Integer.parseInt(numStr[s.length() - i - 1]);
        }
    }

    // compare positive integers
    private boolean larger(BigInteger big) {
        if (big.num.length > num.length) {
            return true;
        } else if (big.num.length < num.length) {
            return false;
        } else {
            int length = num.length;
            for (int i = 0; i < length; i++) {
                if (big.num[length - i - 1] > num[length - i - 1]) {
                    return true;
                } else if (big.num[length - i - 1] < num[length - i - 1]) {
                    return false;
                }
            }
            return false;
        }
    }

    // always positive inputs
    public BigInteger add(BigInteger big) {
        BigInteger larger, smaller;
        if (larger(big)) {
            larger = big;
            smaller = this;
        } else {
            larger = this;
            smaller = big;
        }

        int len = larger.num.length;
        int[] newNum = new int[len + 1];
        int carry = 0;

        for (int i = 0; i < len; i++) {
            int sum;
            if (i >= smaller.num.length) {
                sum = carry + larger.num[i];
            } else {
                sum = carry + larger.num[i] + smaller.num[i];
            }

            if (sum >= 10) {
                String[] nums = Integer.toString(sum).split("");
                newNum[i] = Integer.parseInt(nums[1]);
                carry = Integer.parseInt(nums[0]);
            } else {
                newNum[i] = sum;
                carry = 0;
            }
        }
        newNum[len] = carry;

        return new BigInteger(newNum);
    }

    // always positive inputs
    public BigInteger subtract(BigInteger big) {
        BigInteger larger, smaller;
        boolean sign;
        if (larger(big)) {
            larger = big;
            smaller = this;
            sign = false;
        } else {
            larger = this;
            smaller = big;
            sign = true;
        }

        int[] newNum = new int[larger.num.length];
        int carry = 0;

        for (int i = 0; i < larger.num.length; i++) {
            int sub;
            if (i >= smaller.num.length) {
                sub = larger.num[i] - carry;
            } else {
                sub = larger.num[i] - smaller.num[i] - carry;
            }

            if (sub < 0) {
                newNum[i] = sub + 10;
                carry = 1;
            } else {
                newNum[i] = sub;
                carry = 0;
            }
        }

        BigInteger newBig = new BigInteger(newNum);
        newBig.sign = sign;
        return newBig;
    }

    public BigInteger multiply(BigInteger big) {
        BigInteger newBig = new BigInteger(0);

        for (int i = 0; i < num.length; i++) {
            int[] tempNum = new int[big.num.length + i + 1];
            for (int j = 0; j < i; j++) {
                tempNum[j] = 0;
            }

            int carry = 0;
            for (int j = 0; j < big.num.length; j++) {
                int mul = num[i] * big.num[j] + carry;
                if (mul >= 10) {
                    String[] nums = Integer.toString(mul).split("");
                    tempNum[i + j] = Integer.parseInt(nums[1]);
                    carry = Integer.parseInt(nums[0]);
                } else {
                    tempNum[i + j] = mul;
                    carry = 0;
                }
            }
            tempNum[big.num.length + i] = carry;

            BigInteger tempBig = new BigInteger(tempNum);

            newBig = newBig.add(tempBig);
        }

        newBig.sign = sign == big.sign;

        return newBig;
    }
  
    @Override
    public String toString() {
        String str = "";
        for (int n : num) {
            str = Integer.toString(n) + str;
        }
        if (!sign) str = "-" + str;
        return str;
    }

    static BigInteger evaluate(String input) throws IllegalArgumentException {
        String cleanInput = input.replace(" ", "");
        String firstInput, secondInput;
        char operator;

        Matcher matcher = EXPRESSION_PATTERN.matcher(cleanInput);
        if (matcher.find()) {
            firstInput = matcher.group();
            int index = matcher.end();
            operator = cleanInput.charAt(index);
            secondInput = cleanInput.substring(index + 1);
        } else {
            throw new IllegalArgumentException();
        }

        BigInteger firstBig = new BigInteger(firstInput);
        BigInteger secondBig = new BigInteger(secondInput);
        BigInteger ret = new BigInteger(0);
        switch (operator) {
            case '+':
                if (firstBig.sign) {
                    if (secondBig.sign) ret = firstBig.add(secondBig);
                    else ret = firstBig.subtract(secondBig);
                } else {
                    if (secondBig.sign) ret = secondBig.subtract(firstBig);
                    else {
                        ret = firstBig.add(secondBig);
                        ret.sign = false;
                    }
                }
                break;
            case '-':
                if (firstBig.sign) {
                    if (secondBig.sign) ret = firstBig.subtract(secondBig);
                    else ret = firstBig.add(secondBig);
                } else {
                    if (secondBig.sign) {
                        ret = firstBig.add(secondBig);
                        ret.sign = false;
                    }
                    else ret = secondBig.subtract(firstBig);
                }
                break;
            case '*':
                ret = firstBig.multiply(secondBig);
                break;
        }

        return ret;
    }
  
    public static void main(String[] args) throws Exception {
        try (InputStreamReader isr = new InputStreamReader(System.in))
        {
            try (BufferedReader reader = new BufferedReader(isr))
            {
                boolean done = false;
                while (!done)
                {
                    String input = reader.readLine();
  
                    try
                    {
                        done = processInput(input);
                    }
                    catch (IllegalArgumentException e)
                    {
                        System.err.println(MSG_INVALID_INPUT);
                    }
                }
            }
        }
    }
  
    static boolean processInput(String input) throws IllegalArgumentException
    {
        boolean quit = isQuitCmd(input);
  
        if (quit)
        {
            return true;
        }
        else
        {
            BigInteger result = evaluate(input);
            System.out.println(result.toString());
  
            return false;
        }
    }
  
    static boolean isQuitCmd(String input)
    {
        return input.equalsIgnoreCase(QUIT_COMMAND);
    }
}
