import java.io.*;
import java.util.Scanner;

public class Matching {
	private static MyHashTable<MyString, Range> data;

	public static void main(String args[]) {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		while (true) {
			try {
				String input = br.readLine();
				if (input.compareTo("QUIT") == 0)
					break;

				command(input);
			} catch (IOException e) {
				System.out.println("입력이 잘못되었습니다. 오류 : " + e.toString());
			}
		}
	}

	private static void command(String input) throws IOException {
		int k = 6;
		char commandChar = input.charAt(0);

		if (commandChar == '<') {
			data = new MyHashTable<>(100);
			int line = 0;

			String filename = input.substring(2);
			Scanner scanner = new Scanner(new File(filename));

			while (scanner.hasNext()) {
				line++;
				String str = scanner.nextLine();
				for (int i = 0; i <= str.length() - k; i++) {
					String subStr = str.substring(i, i + k);
					data.insert(new MyString(subStr), new Range(line, i + 1));
				}
			}
		} else if (commandChar == '@') {
			int slot = Integer.parseInt(input.substring(2));
			AVLTree<MyString, Range> tree = data.search(slot);

			if (tree.isEmpty()) {
				System.out.println("EMPTY");
			} else {
				System.out.println(tree.printPreOrderRoot().substring(1));
			}
		} else if (commandChar == '?') {
			String pattern = input.substring(2);
			AVLNode<MyString, Range>[] AVLNodes = new AVLNode[pattern.length() - k + 1];

			for (int i = 0; i <= pattern.length() - k; i++) {
				MyString subPattern = new MyString(pattern.substring(i, i + k));
				AVLTree<MyString, Range> tree = data.search(subPattern.hashCode());
				if (tree.search(subPattern).key == null) {
					System.out.println("(0, 0)");
					return;
				} else {
					AVLNodes[i] = tree.search(subPattern);
				}
			}

			Node<Range> currNode = AVLNodes[0].headValue;
			StringBuilder stringBuilder = new StringBuilder();

			while (currNode.getNext() != null) {
				currNode = currNode.getNext();
				Range prevRange = currNode.getItem();
				boolean continuePoint = false;

				for (int i = 1; i <= pattern.length() - k; i++) {
					Node<Range> tNode = AVLNodes[i].headValue.getNext();

					while (tNode != null && prevRange.compareTo(tNode.getItem()) > 0) {
						tNode = tNode.getNext();
					}

					if (tNode == null || tNode.getItem() == null || prevRange.compareTo(tNode.getItem()) != -1) {
						continuePoint = true;
						break;
					}

					prevRange = tNode.getItem();
				}

				if (continuePoint) {
					continue;
				}

				stringBuilder.append(" ").append(currNode.getItem().toString());
			}

			if (stringBuilder.length() == 0) {
				System.out.println("(0, 0)");
			} else {
				System.out.println(stringBuilder.substring(1));
			}
		} else {
			throw new IOException("지원하지 않는 명령어입니다.");
		}
	}
}

class Range implements Comparable<Range>{
	int line, index;

	public Range(int line, int index) {
		this.line = line;
		this.index = index;
	}

	@Override
	public String toString() {
		return "(" + line + ", " + index + ")";
	}

	@Override
	public int compareTo(Range range) {
		if (line != range.line) {
			return (line - range.line) * 2;
		} else {
			return index - range.index;
		}
	}
}