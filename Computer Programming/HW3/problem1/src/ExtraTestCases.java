import java.util.List;

public class ExtraTestCases {
    public static void main(String[] args) {
        testSubproblem1();
        testSubproblem2();
        testSubproblem3();
        testSubproblem4();
    }

    static MovieApp initializeMovieApp() {
        MovieApp movieApp = new MovieApp();
        movieApp.addMovie("Toy Story", new String[] {"love", "touching"});
        movieApp.addMovie("La La Land", new String[] {"touching", "love"});
        movieApp.addMovie("The Jocker", new String[] {"dark", "touching"});
        movieApp.addMovie("Abatar", new String[] {"love", "touching", "adventure"});
        movieApp.addMovie("Avengers", new String[] {}); // if given tas are empty, do not create Movie object
        movieApp.addUser("Olivia");
        movieApp.addUser("Jack");
        movieApp.addUser("Blue");
        return movieApp;
    }

    static void testSubproblem1() {
        System.out.println("testSubproblem1");
        MovieApp movieApp = initializeMovieApp();
        Movie jocker = movieApp.findMovie("The Jocker");    // return Movie object  with given title
        Movie avengers = movieApp.findMovie("Avengers");    // return null if there's no Movie with given title
        User olivia = movieApp.findUser("Olivia");      // return User object with given username
        System.out.println("\tShould be The Jocker : " + jocker);
        System.out.println("\tShould be null : " + avengers);
        System.out.println("\tShould be Olivia : " + olivia);
        System.out.println("\tShould be false : " +
                movieApp.addUser("Olivia"));    // already exists a user with the given name
        System.out.println("\tShould be false : " + movieApp.addMovie("Toy Story", new String[] {"fun"}));  //  Movie with same title already registered
    }

    static void testSubproblem2() {
        System.out.println("testSubproblem2");
        MovieApp movieApp = initializeMovieApp();
        List<Movie> foundMovies = movieApp.findMoviesWithTags(new String[] {"touching", "love"});
        System.out.println("\tShould be [Toy Story, La La Land, Abatar] : " + foundMovies);
        System.out.println("\tShould be [] : " + movieApp.findMoviesWithTags(new String[] {"hell"}));   // given tags match none of the movies
        System.out.println("\tShould be [] : " + movieApp.findMoviesWithTags(new String[] {})); // empty String array is given as tags
        System.out.println("\tShould be [] : " + movieApp.findMoviesWithTags(null));    // null is given as tags
    }

    static void testSubproblem3() {
        System.out.println("testSubproblem3");
        MovieApp movieApp = initializeMovieApp();
        User olivia = movieApp.findUser("Olivia");
        movieApp.rateMovie(olivia, "The Jocker", 3);
        System.out.println("\tShould be 3 : " + movieApp.getUserRating(olivia, "The Jocker"));
        movieApp.rateMovie(olivia, "The Jocker", 4);
        System.out.println("\tShould be 4 : " + movieApp.getUserRating(olivia, "The Jocker"));  // if a user rates the same movie multiple times, only the last rating is stored
        System.out.println("\tShould be 0 : " + movieApp.getUserRating(olivia, "Toy Story"));   // if the user has not rated the movie, return 0
        System.out.println("\tShould be -1 : " + movieApp.getUserRating(olivia, null)); // if the movie with the title is null, return -1
        System.out.println("\tShould be -1 : " + movieApp.getUserRating(olivia, "Interstellar"));   // if the movie with the title is not registered, return -1
        User mark = movieApp.findUser("Mark");  // if there's no user with the given name, returns null
        movieApp.rateMovie(mark, "The Joker", 2);
        System.out.println("\tShould be -1 : " + movieApp.getUserRating(mark, "The Jocker")); // if the user is null, return -1

    }

    static void testSubproblem4() {
        System.out.println("testSubproblem4");
        MovieApp movieApp = initializeMovieApp();
        User olivia = movieApp.findUser("Olivia"),
                jack = movieApp.findUser("Jack"),
                blue = movieApp.findUser("Blue");
        List<Movie> foundMovies1 = movieApp.findUserMoviesWithTags(olivia, new String[] {"touching", "love"});
        List<Movie> foundMovies2 = movieApp.findUserMoviesWithTags(olivia, new String[] {"dark"});
        System.out.println("\tShould be [Toy Story, La La Land, Abatar] : " + foundMovies1);
        System.out.println("\tShould be [The Jocker] : " + foundMovies2);
        movieApp.rateMovie(olivia, "La La Land", 3);
        movieApp.rateMovie(olivia, "Toy Story", 1);
        movieApp.rateMovie(jack, "La La Land", 4);
        movieApp.rateMovie(jack, "The Jocker", 4);
        movieApp.rateMovie(blue, "Toy Story", 5);
        movieApp.rateMovie(blue, "The Jocker", 2);
        List<Movie> recommendedMovies = movieApp.recommend(olivia); // all movies have same search frequency. sort by median ratings & lexicographical order of titles
        System.out.println("\tShould be [La La Land, The Jocker, Toy Story] : " + recommendedMovies);
        movieApp.findUserMoviesWithTags(olivia, new String[] {"touching"});
        movieApp.findUserMoviesWithTags(olivia, new String[] {"adventure"});
        movieApp.rateMovie(blue, "La La Land", 5);
        movieApp.rateMovie(olivia, "Toy Story", 4); // if a user rates the same movie multiple times, only the last rating is stored
        System.out.println( "\tShould be [Abatar, Toy Story, La La Land] : " +  movieApp.recommend(olivia)); // sort by search frequency & median ratings
        System.out.println("\tShould be [] : " + movieApp.recommend(movieApp.findUser("Jane")));    // if the user is null or not registered, return empty list
        movieApp.addUser("Jane");
        System.out.println("\tShould be [] : " + movieApp.recommend(movieApp.findUser("Jane"))); // if there's no movie to recommend, return empty list
        movieApp.findUserMoviesWithTags(blue, new String[]{"adventure"});
        movieApp.findUserMoviesWithTags(blue, new String[]{"dark"});
        System.out.println("\tShould be [The Joker, Abatar] : " + movieApp.recommend(blue)); // if there are less than 3 movies to recommend, then the returned list may contain less than 3 movies
    }
}