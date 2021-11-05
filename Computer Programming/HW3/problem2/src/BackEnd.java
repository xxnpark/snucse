import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class BackEnd extends ServerResourceAccessible {
    private final String dir = getServerStorageDir();
    private final static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    private static int maxId = 0;

    private Post getPostFromPath(String postPath) {
        File file = new File(postPath);
        int postId = Integer.parseInt(postPath.substring(postPath.lastIndexOf("/") + 1, postPath.lastIndexOf(".")));
        try {
            Scanner sc = new Scanner(file);
            String date = sc.nextLine();
            String title = sc.nextLine();
            sc.nextLine();
            String content = "";
            while (sc.hasNext()) {
                content += sc.nextLine() + "\n";
            }
            return new Post(postId, Post.parseDateTimeString(date, formatter), title, content);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public User auth(String id, String passwdInput) {
        File file = new File(dir);
        String[] userPaths = file.list();
        assert userPaths != null;
        for (String userPath : userPaths) {
            file = new File(dir + userPath + "/post/");
            String[] postPaths = file.list();
            if (postPaths == null) continue;
            for (String postPath : postPaths) {
                int postId = Integer.parseInt(postPath.substring(0, postPath.lastIndexOf(".")));
                if (postId > maxId) {
                    maxId = postId;
                }
            }
        }

        file = new File(dir + id + "/password.txt");
        try {
            Scanner sc = new Scanner(file);
            String passwd = sc.nextLine();
            if (passwd.equalsIgnoreCase(passwdInput)) {
                return new User(id, passwd);
            } else {
                return null;
            }
        } catch (FileNotFoundException e) {
            return null;
        }
    }

    public void post(String title, String content, User user) {
        File file = new File(dir + user.id + "/post/");
        String[] files = file.list();
        assert files != null;

        int postId = files.length == 0 ? 0 : ++maxId;

        try {
            FileWriter fileWriter = new FileWriter(dir + user.id + "/post/" + postId + ".txt");
            Post post = new Post(title, content);
            post.setId(postId);
            fileWriter.write(post.getDate() + "\n");
            fileWriter.write(post.getTitle() + "\n\n");
            fileWriter.write(post.getContent());
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Post> recommend(User user) {
        File file = new File(dir + user.id + "/friend.txt");
        List<String> friends = new LinkedList<>();
        try {
            Scanner sc = new Scanner(file);
            while (sc.hasNext()) {
                friends.add(sc.nextLine());
            }
        } catch (FileNotFoundException e) {
            return null;
        }

        List<Post> friendPosts = new LinkedList<>();
        for (String friend : friends) {
            file = new File(dir + friend + "/post/");
            String[] postPaths = file.list();
            assert postPaths != null;
            for (String postPath : postPaths) {
                String entirePostPath = dir + friend + "/post/" + postPath;
                friendPosts.add(getPostFromPath(entirePostPath));
            }
        }

        friendPosts.sort(new Comparator<Post>() {
            @Override
            public int compare(Post post1, Post post2) {
                return post2.getDate().compareTo(post1.getDate());
            }
        });
        return friendPosts;
    }

    public List<Post> search(Set<String> keywords) {
        List<Post> posts = new LinkedList<>();
        Map<Post, Integer> postKeywordOccurenceMap = new HashMap<>();

        File file = new File(dir);
        String[] userPaths = file.list();
        if (userPaths == null) return null;
        for (String userPath : userPaths) {
            file = new File(dir + userPath + "/post/");
            String[] postPaths = file.list();
            if (postPaths == null) continue;
            for (String postPath : postPaths) {
                String entirePostPath = dir + userPath + "/post/" + postPath;
                Post post = getPostFromPath(entirePostPath);
                if (post == null) continue;
                String data = post.getTitle();
                for (String content : post.getContent().split("\n")) {
                    data += " " + content;
                }

                int occurence = 0;
                for (String word : data.split(" ")) {
                    for (String keyword : keywords) {
                        if (word.equals(keyword)) {
                            occurence++;
                        }
                    }
                }

                posts.add(post);
                postKeywordOccurenceMap.put(post, occurence);
            }
        }

        posts.sort(new Comparator<Post>() {
            @Override
            public int compare(Post post1, Post post2) {
                int count1 = post1.getContent().split("\\s+").length;
                int count2 = post2.getContent().split("\\s+").length;
                return count2 - count1;
            }
        });
        posts.sort(new Comparator<Post>() {
            @Override
            public int compare(Post post1, Post post2) {
                int count1 = postKeywordOccurenceMap.get(post1);
                int count2 = postKeywordOccurenceMap.get(post2);
                return count2 - count1;
            }
        });
        return posts;
    }
}
