package main

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// TokenType
const (
	INTEGER   = "INTERGER"
	REAL      = "REAL"
	PLUS      = "PLUS"
	MINUS     = "MINUS"
	MUL       = "MUL"
	DIV       = "DIV"
	LPAREN    = "("
	RPAREN    = ")"
	ID        = "ID"
	VAR       = "VAR"
	PROGRAM   = "PROGRAM"
	PROCEDURE = "PROCEDURE"
	BEGIN     = "BEGIN"
	END       = "END"
	ASSIGN    = "ASSIGN"
	SEMI      = "SEMI"
	DOT       = "DOT"
	COLON     = "COLON"
	COMMA     = "COMMA"
	EOF       = "EOF"
)

// Token token
type Token struct {
	Type  string
	Value string
}

func newToken(tokenType string, value string) *Token {
	return &Token{
		Type:  tokenType,
		Value: value,
	}
}

var reservedKeyWords = map[string]*Token{
	"PROGRAM":   newToken(PROGRAM, "PROGRAM"),
	"VAR":       newToken(VAR, "VAR"),
	"BEGIN":     newToken(BEGIN, "BEGIN"),
	"END":       newToken(END, "END"),
	"DIV":       newToken(DIV, "DIV"),
	"INTEGER":   newToken(INTEGER, "INTERGER"),
	"REAL":      newToken(REAL, "REAL"),
	"PROCEDURE": newToken("PROCEDURE", "PROCEDURE"),
}

// Lexer lexer
type Lexer struct {
	text string
	pos  int
}

func newLexer(text string) *Lexer {
	return &Lexer{
		text: text,
		pos:  0,
	}
}

func (l *Lexer) skipWhiteSpace() {
	for l.pos < len(l.text) && (l.text[l.pos] == ' ' || l.text[l.pos] == '\t' || l.text[l.pos] == '\n') {
		l.pos++
	}
}

func (l *Lexer) getNextToken() *Token {
	l.skipWhiteSpace()
	if l.pos >= len(l.text) {
		return newToken(EOF, "")
	}

	c := l.text[l.pos]
	for c == '{' {
		for l.pos++; l.text[l.pos] != '}'; l.pos++ {
		}
		l.pos++
		l.skipWhiteSpace()
		if l.pos >= len(l.text) {
			return newToken(EOF, "")
		}
		c = l.text[l.pos]
	}
	switch {
	case c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c == '_':
		begin := l.pos
		for l.pos++; l.pos < len(l.text) && (l.text[l.pos] >= 'A' && l.text[l.pos] <= 'Z' ||
			l.text[l.pos] >= 'a' && l.text[l.pos] <= 'z') || l.text[l.pos] >= '0' && l.text[l.pos] <= '9'; l.pos++ {
		}
		word := l.text[begin:l.pos]
		rword, ok := reservedKeyWords[strings.ToUpper(word)]
		if ok {
			return rword
		}
		return newToken(ID, word)
	case c >= '0' && c <= '9':
		begin := l.pos
		for l.pos++; l.pos < len(l.text) && l.text[l.pos] >= '0' && l.text[l.pos] <= '9'; l.pos++ {
		}
		if l.pos < len(l.text) && l.text[l.pos] == '.' {
			l.pos++
			for l.pos++; l.pos < len(l.text) && l.text[l.pos] >= '0' && l.text[l.pos] <= '9'; l.pos++ {
			}
			return newToken(REAL, l.text[begin:l.pos])
		}
		return newToken(INTEGER, l.text[begin:l.pos])
	case c == '+':
		l.pos++
		return newToken(PLUS, "+")
	case c == '-':
		l.pos++
		return newToken(MINUS, "-")
	case c == '*':
		l.pos++
		return newToken(MUL, "*")
	case c == '/':
		l.pos++
		return newToken(DIV, "/")
	case c == '(':
		l.pos++
		return newToken(LPAREN, "(")
	case c == ')':
		l.pos++
		return newToken(RPAREN, ")")
	case c == ':':
		if l.text[l.pos+1] == '=' {
			l.pos += 2
			return newToken(ASSIGN, ":=")
		}
		l.pos++
		return newToken(COLON, ":")
	case c == ';':
		l.pos++
		return newToken(SEMI, ";")
	case c == '.':
		l.pos++
		return newToken(DOT, ".")
	case c == ',':
		l.pos++
		return newToken(COMMA, ",")
	default:
		panic("getToken:" + string(c))
	}
}

// AST ast
type AST interface {
}

// BinOp op
type BinOp struct {
	left  AST
	token *Token
	right AST
}

func newBinOp(left AST, op *Token, right AST) *BinOp {
	return &BinOp{
		left:  left,
		token: op,
		right: right,
	}
}

// Num number
type Num struct {
	token *Token
	value int
}

func newNum(token *Token) *Num {
	v, error := strconv.Atoi(token.Value)
	if error != nil {
		panic("tonumber")
	}
	return &Num{
		token: token,
		value: v,
	}
}

// RealNum float
type RealNum struct {
	token *Token
	value float64
}

func newRealNum(token *Token) *RealNum {
	v, error := strconv.ParseFloat(token.Value, 64)
	if error != nil {
		panic("real tonumber")
	}
	return &RealNum{
		token: token,
		value: v,
	}
}

// UnaryOp unary
type UnaryOp struct {
	token *Token
	expr  AST
}

func newUnaryOp(token *Token, expr AST) *UnaryOp {
	return &UnaryOp{
		token: token,
		expr:  expr,
	}
}

// Compound compound
type Compound struct {
	children []AST
}

func newCompund() *Compound {
	return &Compound{
		children: make([]AST, 0),
	}
}

type noOp struct {
}

func newNoOp() *noOp {
	return &noOp{}
}

type variable struct {
	token *Token
}

func newVariable(token *Token) *variable {
	return &variable{
		token: token,
	}
}

type assign struct {
	token *Token
	left  *variable
	right AST
}

func newAssign(left *variable, token *Token, right AST) *assign {
	return &assign{
		token: token,
		left:  left,
		right: right,
	}
}

type varDecl struct {
	varNode  []*variable
	typeNode *varType
}

func newVarDecl(node []*variable, typeNode *varType) *varDecl {
	return &varDecl{
		varNode:  node,
		typeNode: typeNode,
	}
}

type varType struct {
	token *Token
}

func newVarType(token *Token) *varType {
	return &varType{
		token: token,
	}
}

type procedureDecl struct {
	name   string
	params []*param
	block  *block
}

func newProdegureDecl(name string, params []*param, block *block) *procedureDecl {
	return &procedureDecl{
		name:   name,
		params: params,
		block:  block,
	}
}

type param struct {
	varNode  *variable
	typeNode *varType
}

func newParam(varNode *variable, typeNode *varType) *param {
	return &param{
		varNode:  varNode,
		typeNode: typeNode,
	}
}

type block struct {
	declarations      []AST
	compoundStatement *Compound
}

func newBlock(decl []AST, compound *Compound) *block {
	return &block{
		declarations:      decl,
		compoundStatement: compound,
	}
}

type program struct {
	name  string
	block *block
}

func newProgram(name string, block *block) *program {
	return &program{
		name:  name,
		block: block,
	}
}

// Parser parser
type Parser struct {
	lexer        *Lexer
	currentToken *Token
}

func newParser(lexer *Lexer) *Parser {
	return &Parser{
		lexer:        lexer,
		currentToken: lexer.getNextToken(),
	}
}

func (p *Parser) eat(tokenType string) {
	if tokenType == p.currentToken.Type {
		p.currentToken = p.lexer.getNextToken()
	} else {
		panic("eat " + tokenType + " " + p.currentToken.Type)
	}
}

func (p *Parser) factor() AST {
	token := p.currentToken
	switch token.Type {
	case LPAREN:
		p.eat(LPAREN)
		node := p.expr()
		p.eat(RPAREN)
		return node
	case INTEGER:
		p.eat(INTEGER)
		node := newNum(token)
		return node
	case REAL:
		p.eat(REAL)
		node := newRealNum(token)
		return node
	case PLUS:
		p.eat(PLUS)
		node := newUnaryOp(token, p.factor())
		return node
	case MINUS:
		p.eat(MINUS)
		node := newUnaryOp(token, p.factor())
		return node
	case ID:
		node := p.variable()
		return node
	default:
		panic("factor error:" + token.Type)
	}
}

func (p *Parser) term() AST {
	node := p.factor()
	for p.currentToken.Type == MUL || p.currentToken.Type == DIV || p.currentToken.Type == ID && p.currentToken.Value == DIV {
		token := p.currentToken
		p.eat(p.currentToken.Type)
		node = newBinOp(node, token, p.factor())
	}
	return node
}

// expr   : term ((PLUS | MINUS) term)*
// term   : factor ((MUL | DIV) factor)*
// factor : ((PLUS | MINUS))factor | INTEGER | LPAREN expr RPAREN
func (p *Parser) expr() AST {
	node := p.term()
	for p.currentToken.Type == PLUS || p.currentToken.Type == MINUS {
		token := p.currentToken
		p.eat(p.currentToken.Type)
		node = newBinOp(node, token, p.term())
	}
	return node
}

func (p *Parser) variable() *variable {
	node := newVariable(p.currentToken)
	p.eat(ID)
	return node
}

func (p *Parser) assignmenStatement() *assign {
	left := p.variable()
	token := p.currentToken
	p.eat(ASSIGN)
	right := p.expr()
	node := newAssign(left, token, right)
	return node
}

func (p *Parser) empty() *noOp {
	return newNoOp()
}

func (p *Parser) statement() AST {
	switch p.currentToken.Type {
	case BEGIN:
		node := p.compoundStatement()
		return node
	case ID:
		node := p.assignmenStatement()
		return node
	default:
		node := p.empty()
		return node
	}
}

func (p *Parser) statementList() []AST {
	node := p.statement()
	nodes := make([]AST, 0, 1)
	nodes = append(nodes, node)
	for p.currentToken.Type == SEMI {
		p.eat(SEMI)
		node = p.statement()
		nodes = append(nodes, node)
	}
	return nodes
}

func (p *Parser) compoundStatement() *Compound {
	p.eat(BEGIN)
	nodes := p.statementList()
	p.eat(END)
	root := newCompund()
	for _, node := range nodes {
		root.children = append(root.children, node)
	}
	return root
}

func (p *Parser) variableDeclaration() *varDecl {
	varNodes := make([]*variable, 0)
	v := newVariable(p.currentToken)
	p.eat(ID)
	varNodes = append(varNodes, v)
	for p.currentToken.Type == COMMA {
		p.eat(COMMA)

		v = p.variable()
		varNodes = append(varNodes, v)
	}
	p.eat(COLON)
	varType := p.typeSpec()
	node := newVarDecl(varNodes, varType)
	return node
}

func (p *Parser) typeSpec() *varType {
	token := p.currentToken
	if p.currentToken.Type == INTEGER {
		p.eat(INTEGER)
	} else {
		p.eat(REAL)
	}
	node := newVarType(token)
	return node
}

func (p *Parser) formalParameters() []*param {
	params := make([]*param, 0)
	tokens := make([]*Token, 0, 1)
	tokens = append(tokens, p.currentToken)
	p.eat(ID)
	for p.currentToken.Type == COMMA {
		p.eat(COMMA)
		tokens = append(tokens, p.currentToken)
		p.eat(ID)
	}
	p.eat(COLON)
	varType := p.typeSpec()
	for _, v := range tokens {
		varNode := newVariable(v)
		node := newParam(varNode, varType)
		params = append(params, node)
	}
	return params
}

func (p *Parser) formalParameterList() []*param {
	params := make([]*param, 0)
	if p.currentToken.Type == LPAREN {
		p.eat(LPAREN)
		params = append(params, p.formalParameters()...)
		for p.currentToken.Type == SEMI {
			p.eat(SEMI)
			params = append(params, p.formalParameters()...)
		}
		p.eat(RPAREN)
	}
	return params
}

func (p *Parser) declarations() []AST {
	varDecls := make([]AST, 0)
	for p.currentToken.Type == VAR {
		p.eat(VAR)
		for p.currentToken.Type == ID {
			node := p.variableDeclaration()
			p.eat(SEMI)
			varDecls = append(varDecls, node)
		}

		for p.currentToken.Type == PROCEDURE {
			p.eat(PROCEDURE)
			procName := p.currentToken.Value
			p.eat(ID)
			params := p.formalParameterList()
			p.eat(SEMI)
			block := p.block()
			node := newProdegureDecl(procName, params, block)
			varDecls = append(varDecls, node)
			p.eat(SEMI)
		}
	}

	return varDecls
}

func (p *Parser) block() *block {
	declarationNodes := p.declarations()
	compound := p.compoundStatement()
	node := newBlock(declarationNodes, compound)
	return node
}

func (p *Parser) program() AST {
	p.eat(PROGRAM)
	token := p.currentToken
	p.eat(ID)
	p.eat(SEMI)
	block := p.block()
	node := newProgram(token.Value, block)
	p.eat(DOT)
	return node
}

func (p *Parser) parse() AST {
	node := p.program()
	if p.currentToken.Type != EOF {
		panic("parse")
	}
	return node
}

// Interpreter i
type Interpreter struct {
	parser      *Parser
	globalScope map[string]float64
}

func newInterpreter(parser *Parser) *Interpreter {
	return &Interpreter{
		parser:      parser,
		globalScope: make(map[string]float64),
	}
}

func (i *Interpreter) visit(tree AST) float64 {
	switch tree.(type) {
	case *Num:
		num := tree.(*Num)
		return float64(num.value)
	case *RealNum:
		num := tree.(*RealNum)
		return num.value
	case *BinOp:
		bo := tree.(*BinOp)
		left := i.visit(bo.left)
		right := i.visit(bo.right)
		switch bo.token.Type {
		case PLUS:
			return left + right
		case MINUS:
			return left - right
		case MUL:
			return left * right
		case DIV:
			if bo.token.Value == "/" {
				return left / right
			}
			return float64(int(left / right))
		default:
			panic("visit")
		}
	case *UnaryOp:
		uo := tree.(*UnaryOp)
		result := i.visit(uo.expr)
		if uo.token.Type == PLUS {
			return result
		}
		return -result
	case *Compound:
		c := tree.(*Compound)
		for _, v := range c.children {
			i.visit(v)
		}
		return 0
	case *assign:
		a := tree.(*assign)
		left := a.left
		right := i.visit(a.right)
		i.globalScope[left.token.Value] = right
		return 0
	case *noOp:
		return 0
	case *variable:
		node := tree.(*variable)
		v, ok := i.globalScope[node.token.Value]
		if ok {
			return v
		}
		return 0
	case *program:
		node := tree.(*program)
		i.visit(node.block)
		return 0
	case *block:
		node := tree.(*block)
		for _, v := range node.declarations {
			i.visit(v)
		}
		i.visit(node.compoundStatement)
		return 0
	case *varDecl:
		return 0
	case *procedureDecl:
		return 0
	default:
		fmt.Println("type:", reflect.TypeOf(tree))
		panic("interpreter visit type")
	}
}

func (i *Interpreter) interpret(tree AST) float64 {
	return i.visit(tree)
}

// Symbol s
type Symbol interface {
	Name() string
}

// BuiltinTypeSymbol b
type BuiltinTypeSymbol struct {
	name string
}

// NewBuiltinTypeSymbol n
func NewBuiltinTypeSymbol(name string) *BuiltinTypeSymbol {
	return &BuiltinTypeSymbol{
		name: name,
	}
}

// Name name
func (b *BuiltinTypeSymbol) Name() string {
	return b.name
}

// VarSymbol v
type VarSymbol struct {
	name string
	Type Symbol
}

// NewVarSymbol n
func NewVarSymbol(name string, symbol Symbol) *VarSymbol {
	return &VarSymbol{
		name: name,
		Type: symbol,
	}
}

type procedureSymbol struct {
	name   string
	params []Symbol
}

// Name name
func (b *procedureSymbol) Name() string {
	return b.name
}

func newProcedureSymbol(name string) *procedureSymbol {
	return &procedureSymbol{
		name:   name,
		params: make([]Symbol, 0),
	}
}

// Name name
func (b *VarSymbol) Name() string {
	return b.name
}

// SymbolTable s
type SymbolTable struct {
	name           string
	level          int
	enclosingScope *SymbolTable
	symbols        map[string]Symbol
}

// NewSymbolTable n
func NewSymbolTable(name string, level int, enclosingScope *SymbolTable) *SymbolTable {
	s := &SymbolTable{
		name:           name,
		level:          level,
		enclosingScope: enclosingScope,
		symbols:        make(map[string]Symbol),
	}
	s.initBuiltins()
	return s
}

func (s *SymbolTable) initBuiltins() {
	s.insert(NewBuiltinTypeSymbol("INTEGER"))
	s.insert(NewBuiltinTypeSymbol("REAL"))
}

func (s *SymbolTable) insert(symbol Symbol) {
	s.symbols[symbol.Name()] = symbol
}

// Lookup l
func (s *SymbolTable) Lookup(name string) Symbol {
	symbol, ok := s.symbols[name]
	if ok {
		return symbol
	}
	if s.enclosingScope != nil {
		return s.enclosingScope.Lookup(name)
	}
	return nil
}

// LookupCurrentScope l
func (s *SymbolTable) LookupCurrentScope(name string) Symbol {
	symbol, ok := s.symbols[name]
	if ok {
		return symbol
	}
	return nil
}

// SemanticAnalyzer s
type SemanticAnalyzer struct {
	symtab *SymbolTable
}

func newSemanticAnalyzer() *SemanticAnalyzer {
	symtab := NewSymbolTable("zero", 0, nil)
	return &SemanticAnalyzer{
		symtab: symtab,
	}
}

func (s *SemanticAnalyzer) visit(tree AST) {
	switch tree.(type) {
	case *Num:
	case *RealNum:
	case *BinOp:
		bo := tree.(*BinOp)
		s.visit(bo.left)
		s.visit(bo.right)
	case *UnaryOp:
		uo := tree.(*UnaryOp)
		s.visit(uo.expr)
	case *Compound:
		c := tree.(*Compound)
		for _, v := range c.children {
			s.visit(v)
		}
	case *assign:
		a := tree.(*assign)
		left := a.left
		varSymbol := s.symtab.Lookup(left.token.Value)
		if varSymbol == nil {
			panic("no symbol " + left.token.Value)
		}
		s.visit(a.right)
	case *noOp:
	case *variable:
		node := tree.(*variable)
		varName := node.token.Value
		varSymbol := s.symtab.Lookup(varName)
		if varSymbol == nil {
			panic("no symbol " + varName)
		}
	case *program:
		fmt.Println("enter score: global")
		globalScope := NewSymbolTable("global", 1, s.symtab)
		s.symtab = globalScope
		node := tree.(*program)
		s.visit(node.block)
		fmt.Println(s.symtab)
		s.symtab = s.symtab.enclosingScope
		fmt.Println("leave scope: global")
	case *block:
		node := tree.(*block)
		for _, v := range node.declarations {
			s.visit(v)
		}
		s.visit(node.compoundStatement)
	case *varDecl:
		node := tree.(*varDecl)
		for _, v := range node.varNode {
			if s.symtab.LookupCurrentScope(v.token.Value) != nil {
				panic("Duplicate identifier " + v.token.Value)
			}
			typeSymbol := s.symtab.Lookup(node.typeNode.token.Value)
			varSymbol := NewVarSymbol(v.token.Value, typeSymbol)
			s.symtab.insert(varSymbol)
		}
	case *procedureDecl:
		node := tree.(*procedureDecl)
		procSymbol := newProcedureSymbol(node.name)
		s.symtab.insert(procSymbol)

		fmt.Println("ENTER scope: " + node.name)
		procedureScope := NewSymbolTable(node.name, s.symtab.level+1, s.symtab)
		s.symtab = procedureScope

		for _, v := range node.params {
			paramType := procedureScope.Lookup(v.typeNode.token.Value)
			paramName := v.varNode.token.Value
			varSymbol := NewVarSymbol(paramName, paramType)
			procedureScope.insert(varSymbol)
			procSymbol.params = append(procSymbol.params, varSymbol)
		}
		s.visit(node.block)
		fmt.Println(procedureScope)
		s.symtab = s.symtab.enclosingScope
		fmt.Println("leave scope: " + node.name)

	default:
		fmt.Println("type:", reflect.TypeOf(tree))
		panic("symbol visit type")
	}
}

func main() {
	text := `
		PROGRAM Part10;
		VAR y: real;
		BEGIN {Part10}
		   y := 20 / 7 + 3.14;
		END. {Part10}
	`
	//text := "20/4"
	//tree := parser.expr()

	lexer := newLexer(text)

	parser := newParser(lexer)
	tree := parser.parse()

	semantic := newSemanticAnalyzer()
	semantic.visit(tree)

	interpreter := newInterpreter(parser)
	result := interpreter.interpret(tree)

	fmt.Println(result, interpreter.globalScope)
}
