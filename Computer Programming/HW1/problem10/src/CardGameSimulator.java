public class CardGameSimulator {
	private static final Player[] players = new Player[2];

	public static void simulateCardGame(String inputA, String inputB) {
		// DO NOT change the skeleton code.
		// You can add codes anywhere you want.

		players[0] = new Player("A", inputA);
		players[1] = new Player("B", inputB);

		Card active = players[0].getDeckCard(0);
		int initialIndex = 0;
		for (int i = 1; i < 10; i++) {
			Card current = players[0].getDeckCard(i);
			if (current.getNumber() > active.getNumber() || current.getNumber() == active.getNumber() && current.getShape() == 'X') {
				active = current;
				initialIndex = i;
			}
		}
		players[0].playCard(active);
		players[0].setDeckNull(initialIndex);

		int turn = 1;

		while (true) {
			boolean continuePoint = false;
			Card maxNumberCard = new Card(-1, 'n');
			int maxNumberIndex = -1;

			for (int i = 0; i < 10; i++) {
				Card current = players[turn].getDeckCard(i);
				if (current == null) continue;

				if (current.getNumber() == active.getNumber()) {
					players[turn].playCard(current);
					players[turn].setDeckNull(i);
					active = current;
					turn = 1 - turn;
					continuePoint = true;
					break;
				} else if (current.getShape() == active.getShape()) {
					if (current.getNumber() > maxNumberCard.getNumber()) {
						maxNumberCard = current;
						maxNumberIndex = i;
					}
				}
			}

			if (continuePoint) continue;

			if (maxNumberCard.getShape() != 'n') {
				players[turn].playCard(maxNumberCard);
				players[turn].setDeckNull(maxNumberIndex);
				active = maxNumberCard;
				turn = 1 - turn;
			} else {
				printWinMessage(players[1 - turn]);
				break;
			}
		}
	}

	private static void printWinMessage(Player player) {
		System.out.printf("Player %s wins the game!\n", player);
	}
}


class Player {
	private String name;
	private Card[] deck;

	Player(String name, String deck) {
		this.name = name;
		this.deck = new Card[10];
		for (int i = 0; i < 10; i++) {
			this.deck[i] = new Card(deck.charAt(3*i) - '0', deck.charAt(3*i+1));
		}
	}

	public void playCard(Card card) {
		System.out.printf("Player %s: %s\n", name, card);
	}

	@Override
	public String toString() {
		return name;
	}

	public Card getDeckCard(int i) {
		return deck[i];
	}

	public void setDeckNull(int i) {
		this.deck[i] = null;
	}
}


class Card {
	private int number;
	private char shape;

	Card(int number, char shape) {
		this.number = number;
		this.shape = shape;
	}

	@Override
	public String toString() {
		return "" + number + shape;
	}

	public int getNumber() {
		return number;
	}

	public char getShape() {
		return shape;
	}
}
