import java.util.Arrays;

public class MoreTests {
    public static void main(String[] args) {
        testSubproblem1();
        testSubproblem2();
        testSubproblem3();
        testSubproblem4();
    }

    public static void check(Object res, Object ans) {
        String resStr = res == null ? "null" : res.toString();
        String ansStr = ans == null ? "null" : ans.toString();
        System.out.println("\tShould be " + ansStr + " : " + resStr);
        assert(resStr.equals(ansStr));
    }

    static void testSubproblem1() {
        System.out.println("testSubproblem1");
        MovieApp movieApp = new MovieApp();
        // addMovie
        {
            // created and stored successfully
            check(movieApp.addMovie("movie1", new String[]{"tag1", "tag2"}), true);
            // the given tags are empty
            check(movieApp.addMovie("no tag", new String[]{}), false);
            // same title was already registered
            check(movieApp.addMovie("movie1", new String[]{"tag3", "tag4"}), false);
        }
        // addUser
        {
            // created and stored successfully
            check(movieApp.addUser("user1"), true);
            check(movieApp.addUser("user2"), true);
            // already exists
            check(movieApp.addUser("user1"), false);
            check(movieApp.addUser("user2"), false);
        }
        // findMovie
        {
            // returns a Movie object with the given title
            check(movieApp.findMovie("movie1"), "movie1");
            // no movie with the given title
            check(movieApp.findMovie("movie2"), null);
            // the given tags are empty
            check(movieApp.findMovie("no tag"), null);
        }
        // findUser
        {
            // returns a User object with the given username
            check(movieApp.findUser("user1"), "user1");
            check(movieApp.findUser("user2"), "user2");
            // no user with the given username
            check(movieApp.findUser("not registered"), null);
        }
    }

    static void testSubproblem2() {
        System.out.println("testSubproblem2");
        MovieApp movieApp = new MovieApp();
        // add movies
        {
            check(movieApp.addMovie("e movie", new String[]{"tag1"}), true);
            check(movieApp.addMovie("a movie", new String[]{"tag2"}), true);
            check(movieApp.addMovie("c movie", new String[]{"tag3"}), true);
            check(movieApp.addMovie("d movie", new String[]{"tag1", "tag2"}), true);
            check(movieApp.addMovie("g movie", new String[]{"tag2", "tag3"}), true);
            check(movieApp.addMovie("f movie", new String[]{"tag3", "tag1"}), true);
            check(movieApp.addMovie("h movie", new String[]{"tag1", "tag2", "tag3"}), true);
            // duplicated tags (NOT specified)
            check(movieApp.addMovie("b movie", new String[]{"tag1", "tag2", "tag3", "tag1", "tag2"}), true);
        }
        // findMoviesWithTags
        {
            check(movieApp.findMoviesWithTags(new String[]{"tag1"}), Arrays.asList("h movie", "f movie", "e movie", "d movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag2"}), Arrays.asList("h movie", "g movie", "d movie", "b movie", "a movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag3"}), Arrays.asList("h movie", "g movie", "f movie", "c movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag1", "tag2"}), Arrays.asList("h movie", "d movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag2", "tag3"}), Arrays.asList("h movie", "g movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag3", "tag1"}), Arrays.asList("h movie", "f movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag1", "tag2", "tag3"}), Arrays.asList("h movie", "b movie"));
            // the empty String array or null is given as a tags argument
            check(movieApp.findMoviesWithTags(new String[]{}), Arrays.asList());
            check(movieApp.findMoviesWithTags(null), Arrays.asList());
            // duplicated tags (NOT specified)
            check(movieApp.findMoviesWithTags(new String[]{"tag2", "tag3", "tag3", "tag2"}), Arrays.asList("h movie", "g movie", "b movie"));
            check(movieApp.findMoviesWithTags(new String[]{"tag1", "tag1", "tag1"}), Arrays.asList("h movie", "f movie", "e movie", "d movie", "b movie"));
        }
    }

    static void testSubproblem3() {
        System.out.println("testSubproblem3");
        MovieApp movieApp = new MovieApp();
        // add users & movies
        {
            check(movieApp.addUser("user a"), true);
            check(movieApp.addUser("user b"), true);
            check(movieApp.addMovie("movie a", new String[]{"tag1"}), true);
            check(movieApp.addMovie("movie b", new String[]{"tag2"}), true);
        }
        User userA = movieApp.findUser("user a");
        User userB = movieApp.findUser("user b");
        {
            check(userA, "user a");
            check(userB, "user b");
        }
        // rateMovie
        {
            // store the rating information and return true.
            check(movieApp.rateMovie(userA, "movie a", 3), true);
            // the title is null, or there is no movie with the given title
            check(movieApp.rateMovie(userA, null, 3), false);
            check(movieApp.rateMovie(userA, "movie c", 3), false);
            // the user is null or not registered
            check(movieApp.rateMovie(null, "movie a", 3), false);
            check(movieApp.rateMovie(new User("yeah"), "movie a", 3), false);
            // the given rating is out of range
            check(movieApp.rateMovie(userA, "movie a", 0), false);
            check(movieApp.rateMovie(userA, "movie a", 6), false);
        }
        // getUserRating
        {
            // returns the rating of the user for the movie with the given title
            check(movieApp.rateMovie(userA, "movie a", 3), true);
            check(movieApp.rateMovie(userB, "movie a", 4), true);
            check(movieApp.rateMovie(userB, "movie b", 2), true);
            check(movieApp.getUserRating(userA, "movie a"), 3);
            check(movieApp.getUserRating(userB, "movie a"), 4);
            check(movieApp.getUserRating(userB, "movie b"), 2);
            // if a user rates the same movie multiple times, only the last rating is stored
            check(movieApp.rateMovie(userB, "movie a", 1), true);
            check(movieApp.rateMovie(userB, "movie a", 5), true);
            check(movieApp.rateMovie(userB, "movie a", 0), false);
            check(movieApp.getUserRating(userB, "movie a"), 5);
            // the user or the movie with the title is null or not registered
            check(movieApp.getUserRating(null, "movie a"), -1);
            check(movieApp.getUserRating(new User("yeah"), "movie a"), -1);
            check(movieApp.getUserRating(userA, null), -1);
            check(movieApp.getUserRating(userA, "yeah"), -1);
            // the user has not rated the movie
            check(movieApp.getUserRating(userA, "movie b"), 0);
        }
    }

    static void testSubproblem4() {
        System.out.println("testSubproblem4");
        // findUserMoviesWithTags
        {
            MovieApp movieApp = new MovieApp();
            check(movieApp.addUser("user a"), true);
            check(movieApp.addUser("user b"), true);
            check(movieApp.addMovie("movie a", new String[]{"tag1", "tag3"}), true);
            check(movieApp.addMovie("movie b", new String[]{"tag2", "tag3"}), true);
            User userA = movieApp.findUser("user a");
            User userB = movieApp.findUser("user b");
            check(movieApp.findUserMoviesWithTags(userA, new String[]{"tag1"}), Arrays.asList("movie a"));
            check(movieApp.findUserMoviesWithTags(userB, new String[]{"tag3"}), Arrays.asList("movie b", "movie a"));
            // if the user is null or not registered, do nothing and return the empty list
            check(movieApp.findUserMoviesWithTags(null, new String[]{"tag1"}), Arrays.asList());
            check(movieApp.findUserMoviesWithTags(new User("yeah"), new String[]{"tag1"}), Arrays.asList());
        }
        // recommend
        {
            MovieApp movieApp = new MovieApp();
            check(movieApp.addUser("user a"), true);
            check(movieApp.addUser("user b"), true);
            check(movieApp.addMovie("movie a", new String[]{"tag1", "tag3"}), true);
            check(movieApp.addMovie("movie b", new String[]{"tag2", "tag3"}), true);
            check(movieApp.addMovie("movie c", new String[]{"tag1", "tag4"}), true);
            check(movieApp.addMovie("movie d", new String[]{"tag2", "tag4"}), true);
            User userA = movieApp.findUser("user a");
            User userB = movieApp.findUser("user b");
            // same number of occurrence, not rated
            check(movieApp.findUserMoviesWithTags(userA, new String[]{"tag1"}), Arrays.asList("movie c", "movie a"));
            check(movieApp.findUserMoviesWithTags(userB, new String[]{"tag3"}), Arrays.asList("movie b", "movie a"));
            check(movieApp.recommend(userA), Arrays.asList("movie a", "movie c"));
            check(movieApp.recommend(userB), Arrays.asList("movie a", "movie b"));
            // same number of occurrence, equal rating
            movieApp.rateMovie(userB, "movie a", 3);
            movieApp.rateMovie(userA, "movie c", 4);
            movieApp.rateMovie(userB, "movie a", 5);
            movieApp.rateMovie(userA, "movie a", 3);
            check(movieApp.recommend(userA), Arrays.asList("movie a", "movie c"));
            // same number of occurrence, diffrent rating
            movieApp.rateMovie(userA, "movie a", 1);
            check(movieApp.recommend(userA), Arrays.asList("movie c", "movie a"));
            // different number of occurrence
            check(movieApp.findUserMoviesWithTags(userA, new String[]{"tag3"}), Arrays.asList("movie b", "movie a"));
            check(movieApp.recommend(userA), Arrays.asList("movie a", "movie c", "movie b"));
            check(movieApp.findUserMoviesWithTags(userA, new String[]{"tag4"}), Arrays.asList("movie d", "movie c"));
            check(movieApp.recommend(userA), Arrays.asList("movie c", "movie a", "movie b"));
            check(movieApp.findUserMoviesWithTags(userA, new String[]{"tag4", "tag2"}), Arrays.asList("movie d"));
            check(movieApp.recommend(userA), Arrays.asList("movie c", "movie a", "movie d"));
            // if the user is null or not registered, return the empty list
            check(movieApp.recommend(null), Arrays.asList());
            check(movieApp.recommend(new User("yeah")), Arrays.asList());
        }
        // more tests on rating
        {
            MovieApp movieApp = new MovieApp();
            check(movieApp.addUser("user a"), true);
            check(movieApp.addUser("user b"), true);
            check(movieApp.addUser("user c"), true);
            check(movieApp.addUser("user d"), true);
            check(movieApp.addUser("user e"), true);
            check(movieApp.addMovie("movie a", new String[]{"tag"}), true);
            check(movieApp.addMovie("movie b", new String[]{"tag"}), true);
            check(movieApp.addMovie("movie c", new String[]{"tag"}), true);
            check(movieApp.addMovie("movie d", new String[]{"tag"}), true);
            check(movieApp.addMovie("movie e", new String[]{"tag"}), true);
            User userA = movieApp.findUser("user a");
            User userB = movieApp.findUser("user b");
            User userC = movieApp.findUser("user c");
            User userD = movieApp.findUser("user d");
            User userE = movieApp.findUser("user e");

            check(movieApp.findUserMoviesWithTags(userC, new String[]{"tag"}), Arrays.asList("movie e", "movie d", "movie c", "movie b", "movie a"));
            check(movieApp.recommend(userC), Arrays.asList("movie a", "movie b", "movie c"));

            // movie a: [5 3 4 1 5] -> 4
            check(movieApp.rateMovie(userA, "movie a", 5), true);
            check(movieApp.rateMovie(userB, "movie a", 3), true);
            check(movieApp.rateMovie(userC, "movie a", 4), true);
            check(movieApp.rateMovie(userD, "movie a", 1), true);
            check(movieApp.rateMovie(userE, "movie a", 5), true);
            // movie b: [] -> 0
            // movie c: [5 1 1 4] -> 2.5
            check(movieApp.rateMovie(userA, "movie c", 5), true);
            check(movieApp.rateMovie(userC, "movie c", 1), true);
            check(movieApp.rateMovie(userD, "movie c", 1), true);
            check(movieApp.rateMovie(userE, "movie c", 4), true);
            // movie d: [2 3 4 5] -> 3.5
            check(movieApp.rateMovie(userA, "movie d", 2), true);
            check(movieApp.rateMovie(userB, "movie d", 3), true);
            check(movieApp.rateMovie(userC, "movie d", 4), true);
            check(movieApp.rateMovie(userE, "movie d", 5), true);
            // movie e: [2 3 4 1 4] -> 3
            check(movieApp.rateMovie(userA, "movie e", 2), true);
            check(movieApp.rateMovie(userB, "movie e", 3), true);
            check(movieApp.rateMovie(userC, "movie e", 4), true);
            check(movieApp.rateMovie(userD, "movie e", 1), true);
            check(movieApp.rateMovie(userE, "movie e", 4), true);

            check(movieApp.recommend(userC), Arrays.asList("movie a", "movie d", "movie e"));

            // movie b: [3] -> 3
            check(movieApp.rateMovie(userB, "movie b", 3), true);
            check(movieApp.recommend(userC), Arrays.asList("movie a", "movie d", "movie b"));
            // movie b: [3 4] -> 3.5
            check(movieApp.rateMovie(userE, "movie b", 4), true);
            check(movieApp.recommend(userC), Arrays.asList("movie a", "movie b", "movie d"));
            // movie c: [5 1 5 4] -> 4.5
            check(movieApp.rateMovie(userC, "movie c", 5), true);
            check(movieApp.recommend(userC), Arrays.asList("movie c", "movie a", "movie b"));
        }
    }
}