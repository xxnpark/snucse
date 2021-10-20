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

        // TODO Practice 2 - Store the created entry in a file
    }

    public void listEntries() {
        Iterator<DiaryEntry> iterator = diaryEntries.listIterator();
        while (iterator.hasNext()) {
            DiaryEntry currentDiaryEntry = iterator.next();
            DiaryUI.print(currentDiaryEntry.getShortString());
        }

        // TODO Practice 2 - Your list should contain previously stored files
    }

    public void listEntriesTitle() {
        diaryEntries.sort(new Comparator<DiaryEntry>() {
            @Override
            public int compare(DiaryEntry diaryEntry1, DiaryEntry diaryEntry2) {
                return diaryEntry1.getTitle().compareToIgnoreCase(diaryEntry2.getTitle());
            }
        });

        DiaryUI.print("List of entries sorted by the title.");
        listEntries();

        // TODO Practice 2 - Your list should contain previously stored files
    }

    public void listEntriesTitleLength() {
        diaryEntries.sort(new Comparator<DiaryEntry>() {
            @Override
            public int compare(DiaryEntry diaryEntry1, DiaryEntry diaryEntry2) {
                int comp = diaryEntry1.getTitle().compareToIgnoreCase(diaryEntry2.getTitle());
                if (comp != 0) {
                    return diaryEntry1.getTitle().compareToIgnoreCase(diaryEntry2.getTitle());
                } else {
                    return diaryEntry2.getContent().length() - diaryEntry1.getContent().length();
                }
            }
        });

        Iterator<DiaryEntry> iterator = diaryEntries.listIterator();
        while (iterator.hasNext()) {
            DiaryEntry currentDiaryEntry = iterator.next();
            DiaryUI.print(currentDiaryEntry.getShortString() + ", length: " + currentDiaryEntry.getContent().length());
        }

        // TODO Practice 2 - Your list should contain previously stored files
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

        // TODO Practice 2 - Your read should contain previously stored files
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

        // TODO Practice 2 - Delete the file of the entry
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

        // TODO Practice 2 - Your search should contain previously stored files
    }
}
