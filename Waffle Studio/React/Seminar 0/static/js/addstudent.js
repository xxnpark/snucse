function add() {
    // HTML 문서의 input 태그로부터 입력 받기
    const name = document.getElementById('input').value;

    // 입력이 없을 경우 Window Alert 생성
    if (!name) {
        window.alert("이름을 입력해주세요.")
    } else {
        // li Element 생성 및 클래스 지정
        const li = document.createElement("li");
        li.setAttribute('class', "name");

        // TextNode 추가
        const textNode = document.createTextNode(name);
        li.appendChild(textNode);

        // li를 ul에 추가
        document.getElementById('box').appendChild(li);

        // 추가 완료 후 입력창 초기화
        document.getElementById('input').value = null;
    }
}

function enterkey() {
    // Enter 키 입력 시 add() 실행
    if (window.event.keyCode == 13) {
        add()
    }
}