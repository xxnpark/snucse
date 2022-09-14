import java.util.Collections;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Genre, Title 을 관리하는 영화 데이터베이스.
 * 
 * MyLinkedList 를 사용해 각각 Genre와 Title에 따라 내부적으로 정렬된 상태를  
 * 유지하는 데이터베이스이다. 
 */
public class MovieDB {
	private MyLinkedList<MovieDBItem> movieDBItemList;

    public MovieDB() {
    	// HINT: MovieDBGenre 클래스를 정렬된 상태로 유지하기 위한 
    	// MyLinkedList 타입의 멤버 변수를 초기화 한다.

		movieDBItemList = new MyLinkedList<>();
    }

    public void insert(MovieDBItem item) {
        // Insert the given item to the MovieDB.

    	// Printing functionality is provided for the sake of debugging.
        // This code should be removed before submitting your work.
        // System.err.printf("[trace] MovieDB: INSERT [%s] [%s]\n", item.getGenre(), item.getTitle());

		movieDBItemList.add(item);
    }

    public void delete(MovieDBItem item) {
        // Remove the given item from the MovieDB.
    	
    	// Printing functionality is provided for the sake of debugging.
        // This code should be removed before submitting your work.
        // System.err.printf("[trace] MovieDB: DELETE [%s] [%s]\n", item.getGenre(), item.getTitle());

		movieDBItemList.remove(item);
    }

    public MyLinkedList<MovieDBItem> search(String term) {
        // Search the given term from the MovieDB.
        // You should return a linked list of MovieDBItem.
        // The search command is handled at SearchCmd class.
    	
    	// Printing search results is the responsibility of SearchCmd class. 
    	// So you must not use System.out in this method to achieve specs of the assignment.

        // This tracing functionality is provided for the sake of debugging.
        // This code should be removed before submitting your work.
    	// System.err.printf("[trace] MovieDB: SEARCH [%s]\n", term);

		MovieDBList results = new MovieDBList();

		for (MovieDBItem item : movieDBItemList) {
			if (item.getTitle().contains(term)) {
				results.add(item);
			}
		}

		return results;
    }
    
    public MyLinkedList<MovieDBItem> items() {
        // Search the given term from the MovieDatabase.
        // You should return a linked list of QueryResult.
        // The print command is handled at PrintCmd class.

    	// Printing movie items is the responsibility of PrintCmd class. 
    	// So you must not use System.out in this method to achieve specs of the assignment.

    	// Printing functionality is provided for the sake of debugging.
        // This code should be removed before submitting your work.
        // System.err.printf("[trace] MovieDB: ITEMS\n");

    	return movieDBItemList;
    }
}

class MovieDBList extends MyLinkedList<MovieDBItem> {
	Node<MovieDBItem> last = head;

	@Override
	public void add(MovieDBItem item) {
		last.insertNext(item);
		last = last.getNext();
		numItems++;
	}
}

/*
 * Classes unused in service
 * Although used in AssignmentGuide.java
 */

class MovieList implements ListInterface<String> {
	public MovieList() {
	}

	@Override
	public Iterator<String> iterator() {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public boolean isEmpty() {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public int size() {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public void add(String item) {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public void remove(String item) {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public String first() {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public void removeAll() {
		throw new UnsupportedOperationException("not implemented yet");
	}
}

class Genre extends Node<String> implements Comparable<Genre> {
	public Genre(String name) {
		super(name);
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public int compareTo(Genre o) {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public int hashCode() {
		throw new UnsupportedOperationException("not implemented yet");
	}

	@Override
	public boolean equals(Object obj) {
		throw new UnsupportedOperationException("not implemented yet");
	}
}