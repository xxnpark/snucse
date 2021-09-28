public class FractionalNumberCalculator {
	public static void printCalculationResult(String equation) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

		int spaceIndex = equation.indexOf(" ");
		char operator = equation.charAt(spaceIndex + 1);

		FractionalNumber num1 = new FractionalNumber(equation.substring(0, spaceIndex) + "/1");
		FractionalNumber num2 = new FractionalNumber(equation.substring(spaceIndex + 3) + "/1");

		int numerator;
		int denominator = num1.getDenominator() * num2.getDenominator();

		if (operator == '+') {
			numerator = num1.getNumerator() * num2.getDenominator() + num1.getDenominator() * num2.getNumerator();
		} else if (operator == '-') {
			numerator = num1.getNumerator() * num2.getDenominator() - num1.getDenominator() * num2.getNumerator();
		} else if (operator == '*') {
			numerator = num1.getNumerator() * num2.getNumerator();
		} else {
			numerator = num1.getNumerator() * num2.getDenominator();
			denominator = num1.getDenominator() * num2.getNumerator();
		}

		int gcd = gcd(Math.abs(numerator), denominator);
		numerator /= gcd;
		denominator /= gcd;
		if (denominator == 1) System.out.println(numerator);
		else System.out.println(numerator + "/" + denominator);
	}

	public static int gcd(int m, int n) {
		if (n == 0) return m;
		return gcd(n, m % n);
	}
}

class FractionalNumber {
	private int numerator;
	private int denominator;

	FractionalNumber(String numString) {
		int slashIndex = numString.indexOf("/");
		int endIndex = numString.indexOf("/", slashIndex + 1);
		if (endIndex == -1) endIndex = numString.length();

		this.numerator = Integer.parseInt(numString.substring(0, slashIndex));
		this.denominator = Integer.parseInt(numString.substring(slashIndex + 1, endIndex));
	}

	public int getNumerator() {
		return numerator;
	}

	public int getDenominator() {
		return denominator;
	}
}
