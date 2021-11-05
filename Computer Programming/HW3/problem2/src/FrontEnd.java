import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.time.LocalDateTime;

public class FrontEnd {
    private UserInterface ui;
    private BackEnd backend;
    private User user;

    public FrontEnd(UserInterface ui, BackEnd backend) {
        this.ui = ui;
        this.backend = backend;
    }
    
    public boolean auth(String authInfo){
        String id = authInfo.split("\n")[0];
        String passwd = authInfo.split("\n")[1];
        User user = backend.auth(id, passwd);
        if (user == null) {
            return false;
        } else {
            this.user = user;
            return true;
        }
    }

    public void post(Pair<String, String> titleContentPair) {
        String id = titleContentPair.key;
        String content = titleContentPair.value;
        backend.post(id, content, user);
    }
    
    public void recommend(int N){
        List<Post> friendPosts = backend.recommend(user);
        if (friendPosts != null) {
            if (friendPosts.size() < N) {
                for (Post post : friendPosts) {
                    ui.println(post);
                }
            } else {
                for (int i = 0; i < N; i++) {
                    ui.println(friendPosts.get(i));
                }
            }
        }
    }

    public void search(String command) {
        String[] inputs = command.split(" ");
        Set<String> keywords = new HashSet<>(Arrays.asList(inputs).subList(1, inputs.length));
        List<Post> posts = backend.search(keywords);
        if (posts != null) {
            if (posts.size() < 10) {
                for (Post post : posts) {
                    ui.println(post.getSummary());
                }
            } else {
                for (int i = 0; i < 10; i++) {
                    ui.println(posts.get(i).getSummary());
                }
            }
        }
    }
    
    User getUser(){
        return user;
    }
}
