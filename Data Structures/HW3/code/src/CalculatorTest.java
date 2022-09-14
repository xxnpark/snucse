import java.io.*;
import java.util.Stack;

public class CalculatorTest {
	public static void main(String[] args) {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		while (true) {
			try {
				String input = br.readLine();
				if (input.compareTo("q") == 0)
					break;
				command(input);
			} catch (Exception e) {
				// System.out.println("입력이 잘못되었습니다. 오류 : " + e.toString());
				System.out.println("ERROR");
			}
		}
	}

	// 연산자 우선순위 부여
	private static long priority(String operator) throws CalculatorException {
		switch (operator) {
			case "^":
				return 1;
			case "~":
				return 2;
			case "*":
			case "/":
			case "%":
				return 3;
			case "+":
			case "-":
				return 4;
			default:
				throw new CalculatorException();
		}
	}

	// 연산잔에 따른 연산
	private static long calculate(String operator, Stack<Long> numStack) throws CalculatorException {
		switch (operator) {
			case "~": return - numStack.pop();
			case "^": {
				long a = numStack.pop();
				long b = numStack.pop();
				if (b == 0 && a < 0) {
					throw new CalculatorException();
				}
				return (long) Math.pow(b, a);
			}
			case "*": return numStack.pop() * numStack.pop();
			case "/": {
				long a = numStack.pop();
				long b = numStack.pop();
				if (a == 0) {
					throw new CalculatorException();
				}
				return b / a;
			}
			case "%": {
				long a = numStack.pop();
				long b = numStack.pop();
				if (a == 0) {
					throw new CalculatorException();
				}
				return b % a;
			}
			case "+": return numStack.pop() + numStack.pop();
			case "-": return - numStack.pop() + numStack.pop();
			default: throw new CalculatorException();
		}
	}

	private static void command(String input) throws CalculatorException {
		Stack<String> operatorStack = new Stack<>();
		Stack<Long> numStack = new Stack<>();
		Stack<String> postfixStack = new Stack<>();

		boolean prevNum = false;
		boolean prevNumSpace = false;
		boolean prevOp = true;

		for (int i = 0; i < input.length(); i++) {
			char val = input.charAt(i);

			if (Character.isDigit(val)) {
				if (prevNumSpace) {
					throw new CalculatorException();
				}

				if (prevNum) {
					long num = Long.parseLong(postfixStack.pop()) * 10L + val - '0';
					postfixStack.push(Long.toString(num));
					numStack.pop();
					numStack.push(num);
				} else {
					postfixStack.push(Character.toString(val));
					numStack.push((long) (val - '0'));
				}

				prevNum = true;
				prevNumSpace = false;
				prevOp = false;
			} else if (val == ' ' || val == '\t') {
				if (prevNum || prevNumSpace) {
					prevNumSpace = true;
				}

				prevNum = false;
			} else if (val == '(') {
				operatorStack.push(Character.toString(val));

				prevNum = false;
				prevNumSpace = false;
				prevOp = true;
			} else if (val == ')') {
				while (!operatorStack.peek().equals("(")) {
					String operator = operatorStack.pop();
					postfixStack.push(operator);
					numStack.push(calculate(operator, numStack));
				}

				operatorStack.pop();

				prevNum = false;
				prevNumSpace = false;
				prevOp = false;
			} else if (val == '^') {
				while (!operatorStack.isEmpty() && !operatorStack.peek().equals("(") && priority(operatorStack.peek()) < priority(Character.toString(val))) {
					String operator = operatorStack.pop();
					postfixStack.push(operator);
					numStack.push(calculate(operator, numStack));
				}

				operatorStack.push(Character.toString(val));

				prevNum = false;
				prevNumSpace = false;
				prevOp = true;
			} else if (val == '+' || val == '-' || val == '*' || val == '/' || val == '%') {
				if (val == '-' && prevOp) {
					operatorStack.push("~");
				} else {
					while (!operatorStack.isEmpty() && !operatorStack.peek().equals("(") && priority(operatorStack.peek()) <= priority(Character.toString(val))) {
						String operator = operatorStack.pop();
						postfixStack.push(operator);
						numStack.push(calculate(operator, numStack));
					}

					operatorStack.push(Character.toString(val));
				}

				prevNum = false;
				prevNumSpace = false;
				prevOp = true;
			} else {
				throw new CalculatorException();
			}
		}

		while(!operatorStack.isEmpty()) {
			String operator = operatorStack.pop();
			postfixStack.push(operator);
			numStack.push(calculate(operator, numStack));
		}

		if (numStack.size() != 1) {
			throw new CalculatorException();
		}

		StringBuilder postfix = new StringBuilder(postfixStack.pop());
		while (!postfixStack.isEmpty()) {
			postfix.insert(0, postfixStack.pop() + " ");
		}

		System.out.println(postfix);
		System.out.println(numStack.pop());
	}
}

class CalculatorException extends Exception {
	CalculatorException() {
		super("ERROR");
	}
}
