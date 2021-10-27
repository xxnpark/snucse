import java.util.*;

public class Diary {
    private List<DiaryEntry> diaryEntries = new LinkedList<>();
    private Map<Integer, Set<String>> searchMap = new HashMap<>();

	public Diary() {
    	// TODO
	}
	
    public void createEntry() {
        String title = DiaryUI.input("title: ");
        String content = DiaryUI.input("content: ");

        DiaryEntry diaryEntry = new DiaryEntry(title, content);
        diaryEntries.add(diaryEntry);

        Set<String> keywords = new HashSet<>();
        for (String keyword: diaryEntry.getTitle().split("\\s")) {
            keywords.add(keyword);
        }
        for (String keyword: diaryEntry.getContent().split("\\s")) {
            keywords.add(keyword);
        }
        searchMap.put(diaryEntry.getID(), keywords);

        DiaryUI.print("The entry is saved.");
    }

    public void listEntries() {
        for (DiaryEntry currentDiaryEntry : diaryEntries) {
            DiaryUI.print(currentDiaryEntry.getShortString());
        }
    }

    public void listEntries(String condition1) {
        List<DiaryEntry> temp = new ArrayList<>(diaryEntries);
        temp.sort(new titleSort());
        DiaryUI.print("List of entries sorted by " + condition1 + ".");

        for (DiaryEntry currentDiaryEntry : temp) {
            DiaryUI.print(currentDiaryEntry.getShortString());
        }
    }

    public void listEntries(String condition1, String condition2) {
        List<DiaryEntry> temp = new ArrayList<>(diaryEntries);
        temp.sort(new lengthSort());
        temp.sort(new titleSort());
        DiaryUI.print("List of entries sorted by " + condition1 + " and " + condition2 + ".");

        for (DiaryEntry currentDiaryEntry : temp) {
            DiaryUI.print(currentDiaryEntry.getShortString() + ", length: " + currentDiaryEntry.getContent().split("\\s").length);
        }
    }

    private DiaryEntry findEntry(int id) {
        for (DiaryEntry currentDiaryEntry: diaryEntries) {
            if (currentDiaryEntry.getID() == id) {
                return currentDiaryEntry;
            }
        }
        return null;
    }

    public void readEntry(int id) {
        DiaryEntry diaryEntry = findEntry(id);

        if (diaryEntry == null) {
            DiaryUI.print("There is no entry with id " + id + ".");
            return;
        }

        DiaryUI.print(diaryEntry.getFullString());
    }

    public void deleteEntry(int id) {
        DiaryEntry diaryEntry = findEntry(id);

        if (diaryEntry == null) {
            DiaryUI.print("There is no entry with id " + id + ".");
            return;
        }

        diaryEntries.remove(diaryEntry);
        searchMap.remove(id);
        DiaryUI.print("Entry " + id + "is removed.");
    }

    public void searchEntry(String keyword) {
        List<DiaryEntry> searchResult = new ArrayList<>();

        for (int id: searchMap.keySet()) {
            if (searchMap.get(id).contains(keyword)) {
                searchResult.add(findEntry(id));
            }
        }

        if (searchResult.isEmpty()) {
            DiaryUI.print("There is no entry that contains \"" + keyword +  "\".");
            return;
        }

        for (DiaryEntry entry: searchResult) {
            DiaryUI.print(entry.getFullString());
        }
    }
}

class titleSort implements Comparator<DiaryEntry> {
    @Override
    public int compare(DiaryEntry entry1, DiaryEntry entry2) {
        return entry1.getTitle().compareTo(entry2.getTitle());
    }
}

class lengthSort implements Comparator<DiaryEntry> {
    @Override
    public int compare(DiaryEntry entry1, DiaryEntry entry2) {
        int length1 = entry1.getContent().split("\\s").length;
        int length2 = entry2.getContent().split("\\s").length;
        return length2 - length1;
    }
}
