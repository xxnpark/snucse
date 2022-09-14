import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class prac {
    public static void main(String[] args) {
        Pattern ex = Pattern.compile("-?[0-9]+");
        String EXPRESSION = "-?[0-9]+";

        Matcher matcher = ex.matcher("12+-12");
        matcher.find();
        System.out.println(matcher.group());
        System.out.println(matcher.end());
        matcher.find();
        System.out.println(matcher.group());

        System.out.println(Arrays.toString("12+12".split(EXPRESSION)));
        System.out.println(Arrays.toString("-12+12".split(EXPRESSION)));
        System.out.println(Arrays.toString("-12+-12".split(EXPRESSION)));
        System.out.println(Arrays.toString("12*-12".split(EXPRESSION)));
        System.out.println(Arrays.toString("12*12".split(EXPRESSION)));
        System.out.println(Arrays.toString("-12*12".split(EXPRESSION)));
        System.out.println(Arrays.toString("-12*-12".split(EXPRESSION)));
        System.out.println(Arrays.toString("-12-12".split(EXPRESSION)));
    }
}
