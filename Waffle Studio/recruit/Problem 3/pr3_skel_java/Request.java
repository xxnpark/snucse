enum Command {
    ADD, DELETE, LIST, PIN, UNPIN, QUIT
}

enum Option {
    r, o, g, n, a
}

//< @brief String command 를 전달받아서, Application.execute()에서 사용될 수 있도록 적절히 파싱해 두는 클래스
public class Request {
    // TODO implement here: 필드 자유롭게 추가
    
    //< @brief 생성자
    Request(String command) {
        // TODO implement here
    }
}
