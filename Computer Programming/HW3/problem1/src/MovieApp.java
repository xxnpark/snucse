
import java.util.*;

public class MovieApp {

    private int movieId = 0;
    private List<Movie> movieList = new LinkedList<>();
    private Map<String, Integer> movieMap = new HashMap<>();

    private int userId = 0;
    private List<User> userList = new LinkedList<>();
    private Map<String, Integer> userMap = new HashMap<>();

    private Map<User, Map<Movie, Integer>> ratings = new HashMap<>();
    private Map<User, Map<Movie, Integer>> searches = new HashMap<>();

    private boolean checkSubSet(String[] arr1, String[] arr2) {
        for (String val : arr1) {
            if (!Arrays.asList(arr2).contains(val)) {
                return false;
            }
        }
        return true;
    }

    private float medianOfIntegerList(List<Integer> list) {
        if (list.size() == 0) {
            return 0;
        } else if (list.size() % 2 == 0) {
            return (float)(list.get(list.size() / 2) + list.get(list.size() / 2 - 1)) / 2;
        } else {
            return (float)list.get(list.size() / 2);
        }
    }

    public boolean addMovie(String title, String[] tags) {
        if (movieMap.containsKey(title) || tags.length == 0) {
            return false;
        }
        Movie movie = new Movie(title, tags);
        movieList.add(movie);
        movieMap.put(title, movieId++);
        return true;
    }

    public boolean addUser(String name) {
        if (userMap.containsKey(name)) {
            return false;
        }
        User user = new User(name);
        userList.add(user);
        userMap.put(name, userId++);
        return true;
    }

    public Movie findMovie(String title) {
        Integer id = movieMap.get(title);
        if (id == null) {
            return null;
        } else {
            return movieList.get(id);
        }
    }

    public User findUser(String username) {
        Integer id = userMap.get(username);
        if (id == null) {
            return null;
        } else {
            return userList.get(id);
        }
    }

    public List<Movie> findMoviesWithTags(String[] tags) {
        if (tags == null || tags.length == 0) {
            return new LinkedList<Movie>();
        }
        List<Movie> foundMovies = new LinkedList<>();
        for (Movie movie : movieList) {
            if (checkSubSet(tags, movie.getTags())) {
                foundMovies.add(movie);
            }
        }
        foundMovies.sort(new titleSort());
        return foundMovies;
    }

    public boolean rateMovie(User user, String title, int rating) {
        if (title == null || movieMap.get(title) == null || user == null || userMap.get(user.getUsername()) == null || rating < 1 || rating > 5) {
            return false;
        }
        Map<Movie, Integer> userRatings = ratings.get(user);
        Movie movie = findMovie(title);
        if (userRatings != null) {
            userRatings.put(movie, rating);
        } else {
            userRatings = new HashMap<>();
            userRatings.put(movie, rating);
            ratings.put(user, userRatings);
        }
        return true;
    }

    public int getUserRating(User user, String title) {
        if (title == null || movieMap.get(title) == null || user == null || userMap.get(user.getUsername()) == null) {
            return -1;
        }
        Map<Movie, Integer> userRatings = ratings.get(user);
        if (userRatings == null) {
            return 0;
        }
        Movie movie = findMovie(title);
        Integer rating = userRatings.get(movie);
        if (rating == null) {
            return 0;
        } else {
            return rating;
        }
    }

    public List<Movie> findUserMoviesWithTags(User user, String[] tags) {
        if (user == null || userMap.get(user.getUsername()) == null) {
            return new LinkedList<>();
        }
        List<Movie> foundMovies = findMoviesWithTags(tags);
        for (Movie movie : foundMovies) {
            Map<Movie, Integer> userSearches = searches.get(user);
            if (userSearches == null) {
                userSearches = new HashMap<>();
                userSearches.put(movie, 1);
                searches.put(user, userSearches);
            } else {
                userSearches.merge(movie, 1, Integer::sum);
            }
        }
        return foundMovies;
    }

    public List<Movie> recommend(User user) {
        if (user == null || userMap.get(user.getUsername()) == null) {
            return new LinkedList<>();
        }
        Map<Movie, Integer> userSearches = searches.get(user);
        if (userSearches == null) {
            return new LinkedList<>();
        }
        List<Movie> searchedMovies = new LinkedList<>(userSearches.keySet());
        searchedMovies.sort(new titleSort().reversed());
        searchedMovies.sort(new Comparator<>() {
            @Override
            public int compare(Movie movie1, Movie movie2) {
                List<Integer> ratings1 = new LinkedList<>(), ratings2 = new LinkedList<>();
                for (User user : ratings.keySet()) {
                    Integer rating1 = ratings.get(user).get(movie1), rating2 = ratings.get(user).get(movie2);
                    if (rating1 != null) {
                        ratings1.add(rating1);
                    }
                    if (rating2 != null) {
                        ratings2.add(rating2);
                    }
                }
                Collections.sort(ratings1);
                Collections.sort(ratings2);
                float med1 = medianOfIntegerList(ratings1), med2 = medianOfIntegerList(ratings2);
                return (int)((med2 - med1) * 2);
            }
        });
        searchedMovies.sort(new Comparator<>() {
            @Override
            public int compare(Movie movie1, Movie movie2) {
                return userSearches.get(movie2) - userSearches.get(movie1);
            }
        });
        if (searchedMovies.size() < 3) {
            return searchedMovies;
        } else {
            return searchedMovies.subList(0, 3);
        }
    }
}

class titleSort implements Comparator<Movie> {
    @Override
    public int compare(Movie movie1, Movie movie2) {
        return movie2.getTitle().compareTo(movie1.getTitle());
    }
}
